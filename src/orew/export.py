"""Write reports/web/{cohort.json, cases/*.json} for the results page in web/. metrics.json is evaluate.py's job.

Contract: PLAN.md, section "Results frontend". Case files get fusion risk curves and alerts from evaluate.py's p10 alarm
simulator (models/threshold.json): train from the cross-fitted OOF preds, val, test only once evaluate's lock exists.
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from orew import PROCESSED, RAW, REPORTS
from orew.evaluate import EVENTS, HH, HS, LOCK, alarms  # the alert simulator evaluate reports with

OUT = REPORTS / "web"
STEP = 10  # s, case-explorer resolution
SERIES = ["map", "nibp_map", "spo2", "hr", "etco2", "bis", "ppf_rate", "rftn_ug_h"]
PHASES = ["anestart", "opstart", "opend", "aneend"]
DEMO = ["age", "sex", "height", "weight", "bmi", "asa", "emop", "department", "optype", "opname"]


def val(v):
    """JSON-safe scalar: NaN -> None, numpy -> python."""
    return None if pd.isna(v) else v.item() if hasattr(v, "item") else v


def counts(s):
    return {str(k): int(n) for k, n in s.value_counts(dropna=False).items()}


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, allow_nan=False, separators=(",", ":")))


def rftn_ug_h(df):
    """Both remifentanil pumps in ug/h, as clean.py does (20 and 50 ug/mL); older caches lack rftn50_rate."""
    return (20 * df.rftn_rate).add(50 * df.get("rftn50_rate", pd.Series(dtype=float)), fill_value=0)


def labels():
    """labels.py outputs: events table and {subjectid: split}, each None until labels.py has run."""
    ev, sp = PROCESSED / "events.parquet", PROCESSED / "splits.json"
    return (pd.read_parquet(ev) if ev.exists() else None,
            {s: k for k, v in json.loads(sp.read_text()).items() for s in v} if sp.exists() else None)


def sig3(a):
    """3 significant digits, NaN -> None (hypoxemia risks sit near 1 %, where 3 decimals would flatten them)."""
    return [None if np.isnan(v) else float(f"{v:.3g}") for v in a]


def risk_sources(processed=PROCESSED, lock=LOCK):
    """{split: preds file} the page may show: train (cross-fitted OOF), val, and test only after evaluate's locked run."""
    src = {"train": "preds_oof", "val": "preds_val", **({"test": "preds_test"} if lock.exists() else {})}
    return {k: processed / f"{f}.parquet" for k, f in src.items() if (processed / f"{f}.parquet").exists()}


def load_risk(ids, processed=PROCESSED, lock=LOCK):
    """({caseid: preds rows sorted by t_sec}, {(caseid, t_sec): {event: drivers text}}, {event: p10 threshold}) for
    these cases, or None before models.py has run."""
    src, tf = risk_sources(processed, lock), processed / "models" / "threshold.json"
    if not src or not tf.exists():
        return None
    thr = json.loads(tf.read_text())
    assert all(thr[e]["chosen_on"] == "val" for e in EVENTS), "alert threshold must come from val (leakage #10)"
    want = ["caseid", "t_sec", "split", *[f"{x}_{e}" for e in EVENTS for x in ("eligible", "lead_specialist")],
            *[f"p_fusion_{e}_{h}" for e in EVENTS for h in HS]]
    frames, drivers = [], {}
    for k, p in src.items():
        d = pd.read_parquet(p, columns=[c for c in want if c in pq.read_schema(p).names], filters=[("caseid", "in", list(ids))])
        frames.append(d := d[d.split == k])  # a case is scored only from its own split's file
        rf = processed / f"records_{k}.jsonl"  # explain.py writes val / test only, so train alerts have no drivers
        recs = [r for r in map(json.loads, rf.open()) if r["caseid"] in ids] if rf.exists() else []
        # fresh = records' p10 matches these preds (not mtimes: `models --test` rewrites the same val preds)
        pp = {e: dict(zip(zip(d.caseid, d.t_sec), d[f"p_fusion_{e}_{HH}"])) for e in EVENTS}
        if all(math.isclose(r[e]["p10"], pp[e].get((r["caseid"], r["t_sec"]), math.nan), rel_tol=1e-3, abs_tol=1e-6)
               for r in recs for e in EVENTS if r[e]):
            drivers |= {(r["caseid"], r["t_sec"]): {e: r[e] and r[e]["top_drivers_text"] for e in EVENTS} for r in recs}
        else:
            print(f"warning: {rf.name} is from an older model, {k} alerts get no drivers; "
                  f"run `uv run python -m orew.explain --split {k}`, then export again")
    d = pd.concat(frames).sort_values(["caseid", "t_sec"])
    return dict(tuple(d.groupby("caseid"))), drivers, {e: float(thr[e]["threshold"]) for e in EVENTS}


def case_risk(d, t0, thr, drivers):
    """risk + alerts of one case from its preds rows d (sorted by t_sec); p is null where the time is ineligible."""
    t, risk, alerts = d.t_sec.to_numpy(), {}, []
    for e in EVENTS:
        el = d[f"eligible_{e}"].to_numpy(bool)
        p = {h: np.where(el, d[f"p_fusion_{e}_{h}"].to_numpy(float), np.nan) for h in HS}
        risk[e] = {"t_min": ((t - t0) / 60).round(2).tolist(), **{f"p{h}": sig3(p[h]) for h in HS}, "threshold": thr[e]}
        lead = d.get(f"lead_specialist_{e}")  # val / test only
        for i in np.flatnonzero(alarms(p[HH], d.caseid.to_numpy(), t, thr[e])):
            alerts.append({"type": e, "t_min": round((t[i] - t0) / 60, 2), "p10": sig3(p[HH][i:i + 1])[0],
                           "lead_specialist": None if lead is None else val(lead.iat[i]),
                           "top_drivers": drivers.get((int(d.caseid.iat[i]), int(t[i])), {}).get(e)})
    return {"risk": risk, "alerts": sorted(alerts, key=lambda a: a["t_min"])}


def pick(ids, cohort, ev, split_of, n):
    """The n cases to export: up to n // 2 val surgeries with an event first (the explorer opens on the first), then the
    rest; cohort order within each."""
    if ev is None or not split_of:
        return ids[:n]
    subj, has = cohort.set_index("caseid").subjectid, set(ev.caseid)
    first = [c for c in ids if c in has and split_of.get(subj[c]) == "val"][: n // 2]
    return first + [c for c in ids if c not in first][: n - len(first)]


def case_json(df, row, ev=None, split_of=None):
    df = df.assign(rftn_ug_h=rftn_ug_h(df))
    g = df.groupby(-(-df.t_sec // STEP) * STEP)[SERIES].last()  # last finite value in (t - 10, t], like the 2 s grid
    t0 = row.casestart
    out = {
        "caseid": int(row.name),
        "demographics": {k: val(row[k]) for k in DEMO},
        "duration_min": val(round((row.caseend - t0) / 60, 1)),
        "phases_min": {k: val(round((row[k] - t0) / 60, 2)) for k in PHASES},
        "t_min": ((g.index - t0) / 60).round(2).tolist(),
        "series": {c: g[c].round(1).astype(object).where(g[c].notna(), None).tolist() for c in SERIES},
    }
    if split_of:
        out["split"] = split_of.get(row.subjectid)
    # a labelled case with no events has no rows in events.parquet, so "labelled" = labels.py wrote its grid
    if ev is not None and (PROCESSED / "grid" / f"{row.name:04d}.parquet").exists():
        out["events"] = [{"type": r.event, "onset_min": round((r.onset_sec - t0) / 60, 2), "end_min": round((r.end_sec - t0) / 60, 2)}
                         for r in ev[ev.caseid == row.name].itertuples()]
    return out


def export(max_cases=50, out=OUT):
    cases = pd.read_csv(RAW / "cases.csv", encoding="utf-8-sig").set_index("caseid")
    cohort = pd.read_csv(RAW / "cohort.csv")
    ids = [c for c in cohort.caseid if (RAW / f"{c:04d}.parquet").exists()]
    ev, split_of = labels()
    chosen = pick(ids, cohort, ev, split_of, max_cases)
    by_case, drivers, thr = load_risk(set(chosen)) or ({}, {}, {})
    pct_time, has, index = {}, {}, {}
    for c in ids:
        df = pd.read_parquet(RAW / f"{c:04d}.parquet")
        pct_time[c], has[c] = df.drop(columns="t_sec").notna().mean(), df.drop(columns="t_sec").notna().any()
        if c not in chosen:
            continue
        case = case_json(df, cases.loc[c], ev, split_of)
        if c in by_case:
            case.update(case_risk(by_case[c], cases.casestart[c], thr, drivers))
        dump(case, out / "cases" / f"{c}.json")
        index[c] = {"caseid": int(c), "duration_min": case["duration_min"], **case["demographics"], "split": case.get("split")}
    dump([index[c] for c in chosen], out / "cases" / "index.json")

    c = cases.loc[ids]
    age = c.age.quantile([0, 0.25, 0.5, 0.75, 1]).round(1).tolist()
    pt = pd.DataFrame(pct_time)  # signals x cases
    later = {}
    if ev is not None:
        later["n_labelled"] = sum((PROCESSED / "grid" / f"{i:04d}.parquet").exists() for i in ids)
        later["events"] = {e: {"n_events": int(n), "n_cases": int(k)} for e, (n, k) in ev.groupby("event").caseid.agg(["size", "nunique"]).iterrows()}
    if split_of:
        s = cohort.assign(split=cohort.subjectid.map(split_of))  # the split covers the whole cohort, not only cached cases
        later["splits"] = {k: {"n_subjects": int(g.subjectid.nunique()), "n_cases": len(g)} for k, g in s.groupby("split")}
    dump({
        "n_cohort": len(cohort),
        "n_cached": len(ids),
        "n_subjects": int(cohort[cohort.caseid.isin(ids)].subjectid.nunique()),
        "age": dict(zip(["min", "q1", "median", "q3", "max"], map(val, age))),
        "sex": counts(c.sex.fillna("unknown")),
        "asa": counts(c.asa.astype("Int64").astype("string").fillna("unknown")),
        "department": counts(c.department.fillna("unknown")),
        "emergency": {"n": int(c.emop.sum()), "of": len(c)},
        "median_duration_min": val(round(((c.caseend - c.casestart) / 60).median(), 1)),
        "signals": [
            {"name": k, "pct_time": round(t * 100, 1), "pct_cases": round(n * 100, 1)}
            for k, t, n in zip(pt.index, pt.mean(axis=1), pd.DataFrame(has).mean(axis=1))
        ],
        **later,
    }, out / "cohort.json")
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=50, help="case files to export (cohort.json always uses all cached)")
    ids = export(ap.parse_args().max_cases)
    print(f"{len(ids)} cached cases -> {OUT}")
