"""Phase 5 / 10: discrimination, calibration, p10 alarm simulation, value bins, subgroups, drug ablation, all with
patient-bootstrap 95 % CIs. uv run python -m orew.evaluate --split val|test
-> reports/{split}_report.md and reports/web/metrics.json (PLAN.md "Results frontend" contract)."""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from orew import CFG, PROCESSED, RAW, REPORTS, ROOT

EVENTS = ["hypotension", "hypoxemia"]
HS = CFG["windows"]["horizons_min"]
HH = CFG["windows"]["headline_horizon_min"]
WIN = HH * 60  # alert window W, s
STEP = CFG["windows"]["pred_step_s"]
A = CFG["alerts"]
ALL_MODELS = ["baseline1", "baseline2", "snapshot", "snapshot_nodrug", "trajectory", "pattern", "fusion", "fusion_nodrug"]
NODRUG = {"snapshot": "snapshot_nodrug", "fusion": "fusion_nodrug"}
NOW = {"hypotension": "map_now", "hypoxemia": "spo2_now"}
BINS = {"hypotension": CFG["eval"]["map_bins"], "hypoxemia": CFG["eval"]["spo2_bins"]}
DISC = ["auroc", "auprc", "brier", "sens_at_90spec"]
ALERT = ["alarms_per_hour", "false_per_true_alarm", "median_lead_min", "pct_warned_5min", "event_recall"]
ATTRS = ["age_band", "sex", "asa", "department", "emergency"]
LOCK = REPORTS / "test.lock"
CHUNK = 100  # bootstrap columns per pass; bounds memory at ~n x CHUNK float64 per temporary


def boot_counts(n_subjects, B, seed=CFG["seed"]):
    """(B + 1) x S subject draw counts: row 0 is the original sample (all ones), rows 1.. draw subjects with replacement."""
    d = np.random.default_rng(seed).integers(0, n_subjects, (B, n_subjects))
    return np.vstack([np.ones(n_subjects)] + [np.bincount(r, minlength=n_subjects) for r in d]).astype(np.float32)


def disc(p, y, s, Ct):
    """4 x (B + 1) array of auroc, auprc, brier, sens_at_90spec. Ct = boot_counts(...).T, s = subject index per row: a
    resample weights every row by its subject's draw count, so all rows of a subject move together. Tied scores form one
    ROC / PR step (same as sklearn with sample weights)."""
    if not len(p):
        return np.full((4, Ct.shape[1]), np.nan)
    o = np.argsort(-p, kind="stable")
    p, y, s = p[o], y[o].astype(np.float32), s[o]
    g = np.r_[0, np.flatnonzero(np.diff(p)) + 1]  # first row of each tied-score group
    pos = np.add.reduceat(y, g) > 0
    # a run of negative-only groups is one step: tp is flat across it, so auroc / ap / sens@spec are unchanged
    g = g[pos | np.r_[True, pos[:-1]]]
    sq = ((p - y) ** 2).astype(np.float32)
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for c in range(0, Ct.shape[1], CHUNK):
            W = Ct[s, c:c + CHUNK]
            dtp = np.add.reduceat(W * y[:, None], g, axis=0, dtype=np.float64)
            dfp = np.add.reduceat(W, g, axis=0, dtype=np.float64) - dtp
            tp, fp = dtp.cumsum(0), dfp.cumsum(0)
            P, N = tp[-1], fp[-1]
            auroc = (dfp * (tp - dtp / 2)).sum(0) / (P * N)  # each negative: positives scored higher + half the ties
            ap = (dtp * tp / np.maximum(tp + fp, 1e-12)).sum(0) / P
            brier = sq @ W / W.sum(0)
            sens = np.where(fp <= 0.1 * N, tp, 0).max(0) / P
            out.append(np.vstack([auroc, ap, brier, sens]))
    return np.hstack(out)


def tri(v):
    """[estimate, ci_low, ci_high] from v[0] (original sample) and v[1:] (resamples); null when undefined."""
    v = np.asarray(v, float)
    if not np.isfinite(v[0]):
        return None
    b = v[1:][np.isfinite(v[1:])]
    return [float(v[0])] + ([float(x) for x in np.percentile(b, [2.5, 97.5])] if len(b) else [None, None])


def beats(v, b):
    """Model vs baseline 1 on auroc / auprc: point estimate higher, and paired-bootstrap CI of the difference above 0."""
    out = {}
    for i, k in enumerate(["auroc", "auprc"]):
        d = v[i] - b[i]
        bb = d[1:][np.isfinite(d[1:])]
        out[k] = bool(d[0] > 0) if np.isfinite(d[0]) else None
        out[f"{k}_ci"] = bool(len(bb) and np.percentile(bb, 2.5) > 0)
    return out


def reliability(p, y, s, Ct, k=10):
    """k equal-count bins of p: mean predicted p, observed rate (+ its bootstrap CI as obs_ci) and count."""
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in np.array_split(np.argsort(p, kind="stable"), k):
            if len(i):
                W = Ct[s[i]]
                t = tri(y[i].astype(np.float32) @ W / W.sum(0))
                out.append({"p": float(p[i].mean()), "obs": t[0], "n": len(i), "obs_ci": t[1:]})
    return out


def bin_index(x, bins):
    """Index of the [lo, hi) bin each value falls in, -1 outside every bin (or NaN)."""
    out = np.full(len(x), -1)
    for i, (lo, hi) in enumerate(bins):
        out[(x >= lo) & (x < hi)] = i
    return out


def bin_label(ev, lo, hi, last):
    """Contract labels: MAP 65-70 ... >80, SpO2 90-94, >=95 (SpO2 is integer, so [90, 95) reads 90-94)."""
    spo2 = ev == "hypoxemia"
    return (f">={lo}" if spo2 else f">{lo}") if last else f"{lo}-{hi - 1 if spo2 else hi}"


def alarms(p, caseid, t, thr):
    """p10 alert simulator, rows sorted by (caseid, t); NaN never fires. Fires at p >= thr, then stays quiet for
    refractory_s after the last alarm unless p reaches that alarm's p + rearm_delta."""
    out, lc, lt, lp = np.zeros(len(p), bool), None, -np.inf, 0.0
    for i in np.flatnonzero(p >= thr):
        if caseid[i] != lc:
            lc, lt = caseid[i], -np.inf
        if t[i] - lt >= A["refractory_s"] or p[i] >= lp + A["rearm_delta"]:
            out[i], lt, lp = True, t[i], p[i]
    return out


def choose_threshold(d, col):
    """Lowest threshold whose simulated alarm rate on these (VAL) rows is <= alerts.alarms_per_hour (leakage #10).
    Same procedure as models.choose_threshold, so every model alarms at the same rate."""
    d = d.sort_values(["caseid", "t_sec"])
    p, c, t = d[col].to_numpy(), d.caseid.to_numpy(), d.t_sec.to_numpy()
    hours = np.isfinite(p).sum() * STEP / 3600
    cand = np.unique(p[np.isfinite(p)])
    lo, hi = 0, len(cand) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if alarms(p, c, t, cand[mid]).sum() / hours <= A["alarms_per_hour"] else (mid + 1, hi)
    return float(cand[lo])


def alert_metrics(d, p, thr, ev, C):
    """Five alert metrics with CIs (PLAN.md definitions, W = headline horizon). d: monitored rows of one event sorted by
    (caseid, t_sec) with columns caseid, t_sec, s, nxt (s to next onset); p aligned with d. ev: events (caseid,
    onset_sec, s). True alarm: an onset in (t, t + W]. Warned event: an alarm in [onset - W, onset); lead from the first."""
    S = C.shape[1]
    a = alarms(p, d.caseid.to_numpy(), d.t_sec.to_numpy(), thr)
    true = a & (d.nxt.to_numpy() <= WIN)
    per = lambda s, w: np.bincount(s, weights=w, minlength=S)
    s = d.s.to_numpy()
    n_a, n_t, hrs = per(s, a), per(s, true), per(s, np.full(len(d), STEP / 3600))
    at = {c: g.to_numpy() for c, g in d.t_sec[a].groupby(d.caseid[a])}
    lead = np.full(len(ev), np.nan)
    for j, (c, on) in enumerate(zip(ev.caseid, ev.onset_sec)):
        x = at.get(c, np.empty(0))
        i = np.searchsorted(x, on - WIN)
        if i < len(x) and x[i] < on:
            lead[j] = (on - x[i]) / 60
    w, es = np.isfinite(lead), ev.s.to_numpy()
    n_e, n_w, n_w5 = per(es, np.ones(len(ev))), per(es, w), per(es, lead >= 5)
    with np.errstate(invalid="ignore", divide="ignore"):
        tot = lambda v: C @ v
        med = [np.median(np.repeat(lead[w], k[es[w]].astype(int))) if k[es[w]].sum() else np.nan for k in C]
        m = {"alarms_per_hour": tri(tot(n_a) / tot(hrs)), "false_per_true_alarm": tri((tot(n_a) - tot(n_t)) / tot(n_t)),
             "median_lead_min": tri(med), "pct_warned_5min": tri(tot(n_w5) / tot(n_e)), "event_recall": tri(tot(n_w) / tot(n_e))}
    return m, {"alarms": int(a.sum()), "true_alarms": int(true.sum()), "events": len(ev), "warned": int(w.sum()),
               "hours": float(hrs.sum())}


def _job(args):
    """All discrimination blocks for one event x horizon (runs in a worker process)."""
    ev, h, d, models, C = args
    Ct = np.ascontiguousarray(C.T)
    y, s, fl = d.y.to_numpy(), d.s.to_numpy(), d.flag.to_numpy()
    P = {m: d[m].to_numpy() for m in models}
    raw = lambda mask: {m: disc(P[m][mask], y[mask], s[mask], Ct) for m in models}
    summ = lambda r, ks: {m: {k: tri(v[DISC.index(k)]) for k in ks} for m, v in r.items()}
    out = {"overall": summ(raw(np.ones(len(d), bool)), DISC), "overall_excl_flagged": summ(raw(~fl), DISC)}
    bi = bin_index(d.now.to_numpy(), BINS[ev])
    for key, base in [("bins", np.ones(len(d), bool)), ("bins_excl_flagged", ~fl)]:
        out[key] = []
        for i, (lo, hi) in enumerate(BINS[ev]):
            mask = base & (bi == i)
            r = raw(mask)
            ms = summ(r, ["auroc", "auprc"])
            for m in models:
                if m != "baseline1" and "baseline1" in r:
                    ms[m]["beats_baseline1"] = beats(r[m], r["baseline1"])
            out[key].append({"label": bin_label(ev, lo, hi, i == len(BINS[ev]) - 1), "lo": lo, "hi": hi,
                             "n": int(mask.sum()), "n_pos": int(y[mask].sum()), "models": ms})
    out["subgroups"] = {a: [{"label": str(v), "n": int(mask.sum()), "n_pos": int(y[mask].sum()),
                             "models": summ(raw(mask), ["auroc", "auprc"])}
                            for v in sorted(d[a].unique()) for mask in [d[a].to_numpy() == v]] for a in ATTRS}
    out["reliability"] = {m: reliability(P[m], y, s, Ct) for m in models}
    return ev, h, out


def attributes(cases):
    """Subgroup label per caseid: age band, sex, ASA, department, emergency (cases.csv; missing -> 'unknown')."""
    b = CFG["eval"]["age_bands"]
    c = cases.set_index("caseid")
    return pd.DataFrame({
        "age_band": pd.cut(c.age, b, right=False,
                           labels=[f"{lo}-{hi - 1}" for lo, hi in zip(b[:-2], b[1:-1])] + [f">={b[-2]}"]),
        "sex": c.sex, "asa": c.asa.astype("Int64").astype("string"), "department": c.department,
        "emergency": np.where(c.emop == 1, "emergency", "planned")}, index=c.index).astype("string").fillna("unknown")


def evaluate(df, val, events, cases, thr_json, B, split):
    """Everything for one split. df / val: preds parquet frames (val is only used to choose the non-final models'
    alert thresholds); events: events.parquet; cases: cases.csv; thr_json: models/threshold.json or None.
    Returns (metrics dict per the PLAN.md contract, report markdown)."""
    t0 = time.time()
    df = df.sort_values(["caseid", "t_sec"], ignore_index=True)
    models = [m for m in ALL_MODELS if all(f"p_{m}_{e}_{h}" in df for e in EVENTS for h in HS)]
    final = "fusion" if "fusion" in models else [m for m in models if m not in NODRUG.values()][-1]
    subj = np.unique(df.subjectid)
    df["s"] = np.searchsorted(subj, df.subjectid)
    C = boot_counts(len(subj), B)
    att = attributes(cases)
    for a in ATTRS:
        df[a] = df.caseid.map(att[a]).fillna("unknown")
    s_of_case, first_t = df.groupby("caseid").s.first(), df.groupby("caseid").t_sec.min()
    notes, jobs, n, ev_rows = [], [], {}, {}
    for e in EVENTS:
        el = df[f"eligible_{e}"].to_numpy()
        x = events[(events.event == e) & events.caseid.isin(s_of_case.index)].sort_values(["caseid", "onset_sec"])
        # warnable: some prediction time lies in [onset - W, onset); with 30 s steps that is onset > the case's first t
        x = x[x.onset_sec.to_numpy() > first_t[x.caseid].to_numpy()].assign(s=lambda z: s_of_case[z.caseid].to_numpy())
        ev_rows[e] = x
        n[e] = {"cases": int(df.caseid[el].nunique()), "subjects": int(df.s[el].nunique()), "times": int(el.sum()),
                "events": len(x)}
        for h in HS:
            pc = [f"p_{m}_{e}_{h}" for m in models]
            ok = el & df[f"y_{e}_{h}"].notna().to_numpy()
            fin = ok & np.isfinite(df[pc].to_numpy()).all(1)
            if (ok & ~fin).any():
                notes.append(f"{e} {h} min: {(ok & ~fin).sum()} eligible labelled rows dropped (a model's p is NaN).")
            d = df.loc[fin, ["s", f"y_{e}_{h}", f"flag_{e}", NOW[e], *ATTRS, *pc]]
            d.columns = ["s", "y", "flag", "now", *ATTRS, *models]
            jobs.append((e, h, d, models, C))
    res = {k: {e: {} for e in EVENTS} for k in ["overall", "overall_excl_flagged", "bins", "bins_excl_flagged",
                                                "subgroups", "reliability"]}
    with ProcessPoolExecutor(min(len(jobs), os.cpu_count() or 1)) as ex:
        for e, h, out in ex.map(_job, jobs):
            for k, v in out.items():
                res[k][e][str(h)] = v
    t_disc = time.time() - t0
    if split == "val" and final == "fusion":
        notes.append("fusion is isotonic-calibrated on this split, so its val AUROC, AUPRC, Brier and reliability are "
                     "in-sample (optimistic, most of all inside the value bins); the final numbers are the test run.")

    thr = {e: {} for e in EVENTS}
    alert_log = {}
    for e in EVENTS:
        if thr_json and final in models:
            assert thr_json[e]["chosen_on"] == "val", "alert threshold must come from val (leakage #10)"
            thr[e][final] = float(thr_json[e]["threshold"])
        notes.append(f"{e} alert thresholds: " + (f"{final} from models/threshold.json, others " if thr[e] else "all ")
                     + "chosen on val with the same rule (lowest p10 threshold with <= target alarms/h).")
        vrows = val[val[f"eligible_{e}"]]
        for m in models:
            thr[e].setdefault(m, choose_threshold(vrows, f"p_{m}_{e}_{HH}"))
        for key, keep in [("overall", True), ("overall_excl_flagged", False)]:
            mon = df[f"eligible_{e}"] & (keep | ~df[f"flag_{e}"])
            for m in models:
                d = df.loc[mon & df[f"p_{m}_{e}_{HH}"].notna(), ["caseid", "t_sec", "s", f"next_onset_{e}_s"]]
                d.columns = ["caseid", "t_sec", "s", "nxt"]
                am, log = alert_metrics(d, df.loc[d.index, f"p_{m}_{e}_{HH}"].to_numpy(), thr[e][m], ev_rows[e], C)
                res[key][e][str(HH)][m].update(am)
                alert_log[(key, e, m)] = log
    t_all = time.time() - t0

    pair = lambda o, m: {k: o.get(m, {}).get(k) for k in ["auroc", "auprc"]}
    ablation_all = {e: {str(h): {m: {"with_drugs": pair(res["overall"][e][str(h)], m),
                                     "without_drugs": pair(res["overall"][e][str(h)], NODRUG[m])}
                                 for m in NODRUG if m in models and NODRUG[m] in models} for h in HS} for e in EVENTS}
    metrics = {
        "split": split, "final_model": final, "models": [m for m in models if m not in NODRUG.values()],
        "horizons_min": HS, "headline_horizon_min": HH, "bootstrap": B, "n": n,
        "alerts": {"chosen_on": "val", "alarms_per_hour_target": A["alarms_per_hour"], "threshold": thr},
        **res,
        # contract: one model per event x horizon; snapshot (the specialist), fusion's ablation is in the report
        "ablation": {e: {h: {"model": "snapshot", **v["snapshot"]} for h, v in byh.items() if "snapshot" in v}
                     for e, byh in ablation_all.items()},
    }
    timing = f"discrimination + bins + subgroups + reliability {t_disc:.0f} s, alerts {t_all - t_disc:.0f} s"
    return metrics, report(metrics, ablation_all, alert_log, notes, timing)


def f(m, d=3):
    if m is None:
        return "n/a"
    return f"{m[0]:.{d}f} ({m[1]:.{d}f}–{m[2]:.{d}f})" if m[1] is not None else f"{m[0]:.{d}f} (CI n/a)"


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |"
                                                                                 for r in rows]) + "\n"


def report(M, ablation, alert_log, notes, timing):
    sp, B, ms, fm = M["split"], M["bootstrap"], M["models"], M["final_model"]
    L = [f"# Evaluation report: {sp} split\n",
         f"Models: {', '.join(ms)} (final: {fm}). Every metric is estimate (95 % CI) from {B} patient-bootstrap "
         f"resamples (subjects drawn with replacement; all rows of a subject move together). All metrics use all {B} "
         f"resamples. Runtime: {timing}. Only eligible prediction times with a non-censored label are scored.\n"]
    L += [f"- {e}: {v['events']} warnable events in {v['cases']} surgeries ({v['subjects']} patients), "
          f"{v['times']} eligible prediction times." for e, v in M["n"].items()]
    L += [f"- {x}" for x in notes] + [""]
    if M["n"]["hypoxemia"]["events"] < 50:
        L.append("Hypoxemia has few events, so its intervals are wide; read them before the point estimates.\n")
    for flag in ["", "_excl_flagged"]:
        L.append(f"## Overall{' (flagged times dropped)' if flag else ''}\n")
        for e, byh in M[f"overall{flag}"].items():
            L.append(f"### {e}\n")
            L.append(table(["H", "model", "AUROC", "AUPRC", "Brier", "sens @ 90 % spec"],
                           [[h, m, *[f(byh[h][m][k]) for k in DISC]] for h in byh for m in byh[h]]))
    L.append(f"## Alerts (p{HH}, threshold chosen on val at {A['alarms_per_hour']}/h, refractory {A['refractory_s']} s, "
             f"rearm +{A['rearm_delta']})\n")
    L.append(f"True alarm: an onset in (t, t + {HH} min]. Warned event: an alarm in [onset − {HH} min, onset); lead "
             "from the first such alarm. Median lead is over warned events only; the 5-min share is over all events. "
             "Warnable events: onset after the case's first prediction time.\n")
    for key in ["overall", "overall_excl_flagged"]:
        for e in EVENTS:
            L.append(f"### {e}{' (flagged times dropped)' if key != 'overall' else ''}\n")
            o = M[key][e][str(HH)]
            L.append(table(["model", "threshold", "alarms", "true", *ALERT],
                           [[m, f"{M['alerts']['threshold'][e][m]:.4g}", alert_log[(key, e, m)]["alarms"],
                             alert_log[(key, e, m)]["true_alarms"], *[f(o[m].get(k), 2) for k in ALERT]] for m in o]))
    for flag in ["", "_excl_flagged"]:
        L.append(f"## Value bins: model vs baseline1 within the current value{' (flagged times dropped)' if flag else ''}\n")
        L.append("beats b1 = point estimate higher / paired-bootstrap CI of the difference above 0 (AUROC).\n")
        for e, byh in M[f"bins{flag}"].items():
            for h, bins in byh.items():
                L.append(f"### {e}, {h} min, by {NOW[e]}\n")
                L.append(table(["bin", "n", "n_pos", "model", "AUROC", "AUPRC", "beats b1"],
                               [[b["label"], b["n"], b["n_pos"], m, f(v["auroc"]), f(v["auprc"]),
                                 "" if "beats_baseline1" not in v else
                                 f"{v['beats_baseline1']['auroc']} / {v['beats_baseline1']['auroc_ci']}"]
                                for b in bins for m, v in b["models"].items()]))
    L.append(f"## Subgroups ({HH} min, AUROC)\n")
    for e, byh in M["subgroups"].items():
        L.append(f"### {e}\n")
        L.append(table(["attribute", "group", "n", "n_pos", *ms],
                       [[a, g["label"], g["n"], g["n_pos"], *[f(g["models"][m]["auroc"]) for m in ms]]
                        for a, gs in byh[str(HH)].items() for g in gs]))
    L.append("## Drug-feature ablation (with / without drug features)\n")
    L.append(table(["event", "H", "model", "AUROC with", "AUROC without", "AUPRC with", "AUPRC without"],
                   [[e, h, m, f(v["with_drugs"]["auroc"]), f(v["without_drugs"]["auroc"]),
                     f(v["with_drugs"]["auprc"]), f(v["without_drugs"]["auprc"])]
                    for e, byh in ablation.items() for h, bym in byh.items() for m, v in bym.items()]))
    L.append("## Limits\n\n- The claim is \"more time to act\", not \"saves lives\".\n- Results are retrospective and from "
             "a single centre (VitalDB, Seoul National University Hospital); external validation is needed.\n- Prior "
             "work exists (the Hypotension Prediction Index, Prescience). Our contribution is an open build with an "
             "evaluation that rules out the current-value shortcut.\n- Not built yet: trajectory (GRU/CNN) and pattern "
             "models.\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    split = ap.parse_args().split
    if split == "test":
        if LOCK.exists():
            sys.exit(f"{LOCK} exists: the single locked test run has already happened.")
        if subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode:
            sys.exit("pytest failed: refusing to evaluate on test.")
    if (PROCESSED / "events.parquet").stat().st_mtime > (PROCESSED / f"preds_{split}.parquet").stat().st_mtime:
        sys.exit("events.parquet is newer than the predictions: re-run orew.features and orew.models")
    df = pd.read_parquet(PROCESSED / f"preds_{split}.parquet")
    if not (df.split == split).all():
        sys.exit(f"preds_{split}.parquet holds rows from other splits")
    val = df if split == "val" else pd.read_parquet(PROCESSED / "preds_val.parquet")
    tf = PROCESSED / "models" / "threshold.json"
    thr = json.loads(tf.read_text()) if tf.exists() else None
    events = pd.read_parquet(PROCESSED / "events.parquet")
    cases = pd.read_csv(RAW / "cases.csv", encoding="utf-8-sig")
    metrics, md = evaluate(df, val, events, cases, thr, CFG["eval"]["bootstrap"], split)
    (REPORTS / "web").mkdir(parents=True, exist_ok=True)
    (REPORTS / "web" / "metrics.json").write_text(json.dumps(metrics, allow_nan=False, indent=1))
    (REPORTS / f"{split}_report.md").write_text(md)
    if split == "test":
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        LOCK.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit}))
    print(f"wrote reports/{split}_report.md and reports/web/metrics.json")


if __name__ == "__main__":
    main()
