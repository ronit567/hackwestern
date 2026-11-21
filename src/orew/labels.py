"""Phase 2: cleaned grids, events, prediction table, subject splits (uv run python -m orew.labels [--plot-events 20])."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from orew import CFG, PROCESSED, RAW, REPORTS
from orew.clean import FILL, STEP, clean_case

EV = {k: v for k, v in CFG["events"].items() if isinstance(v, dict)}
W = CFG["windows"]


def events(g):
    """One row per event. A run is a maximal stretch of consecutive grid steps whose {signal}_raw is below threshold:
    artifact below threshold is kept at its raw value, artifact at or above it takes the cleaned (bridged) value, so a
    flush spike can't split a run; gaps of <= 30 s are forward-filled, longer gaps stay NaN, and NaN ends a run. Labels
    may see how long a gap lasts, so unlike the causal clean grid the head of a > 30 s gap is not filled. Artifact
    steps at either end of a run are trimmed off (a zero line before the catheter is placed or after it is removed
    would otherwise join the real low MAP next to it), so onset_sec / end_sec are the first / last non-artifact grid
    time of the run, duration_s = steps * STEP (each step stands for its 2 s bin), artifact_frac = share of
    art_{signal} steps in between. A run is an event iff duration_s >= min_duration_s and artifact_frac <= max_artifact_frac."""
    t, out = g.t_sec.to_numpy(), []
    for e, c in EV.items():
        s, thr = c["signal"], c["threshold"]
        flag = g[f"art_{s}"].to_numpy()
        r = g[f"{s}_raw"].mask(flag & (g[f"{s}_raw"] >= thr), g[s])  # a flagged spike takes the bridged value
        v = r.ffill().where(r.notna() | r.isna().groupby(r.notna().cumsum()).transform("sum").le(FILL))
        d = np.diff(np.r_[0, (v < thr).astype(int), 0])
        a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
        real = np.r_[-1, np.flatnonzero(~flag), len(flag)]
        a, b = real[np.searchsorted(real, a)], real[np.searchsorted(real, b, "right") - 1]
        a, b = a[a <= b], b[a <= b]  # an all-artifact run trims to nothing
        art = np.r_[0, np.cumsum(flag)]
        n = b - a + 1
        r = pd.DataFrame({"event": e, "onset_sec": t[a], "end_sec": t[b], "duration_s": n * STEP,
                          "artifact_frac": (art[b + 1] - art[a]) / n})
        out.append(r[(r.duration_s >= c["min_duration_s"]) & (r.artifact_frac <= CFG["events"]["max_artifact_frac"])])
    return pd.concat(out, ignore_index=True)


def prediction_table(g, caseid, subjectid):
    """One row per t = input_s, input_s + 30, ... <= last grid time; ineligible rows are kept with eligible_e False."""
    last = g.t_sec.iat[-1]
    t = np.arange(W["input_s"], last + 1, W["pred_step_s"])
    i, win = t // STEP, W["input_s"] // STEP  # grid row of t; the input window (t - 600, t] is rows i - 299 .. i
    ev, out = events(g), {"caseid": caseid, "subjectid": subjectid, "t_sec": t}
    for e, c in EV.items():
        s = c["signal"]
        bad = np.r_[0, np.cumsum(g[s].isna() | g[f"art_{s}"])]
        x = ev[ev.event == e]
        on, end = x.onset_sec.to_numpy(), x.end_sec.to_numpy()
        j = np.searchsorted(on, t, "right")  # events with onset <= t are on[:j]
        in_progress = np.r_[-np.inf, end][j] >= t
        out[f"eligible_{e}"] = ~in_progress & ((bad[i + 1] - bad[i + 1 - win]) / win <= W["max_missing_frac"])
        out[f"flag_{e}"] = t - np.r_[-np.inf, end][np.searchsorted(end, t)] <= W["flag_after_event_s"]
        nxt = np.r_[on, np.nan][j] - t  # first onset strictly after t
        for h in W["horizons_min"]:
            # censored, not negative, when the label window runs past the recording
            out[f"y_{e}_{h}"] = np.where(t + h * 60 > last, np.nan, nxt <= h * 60)
        out[f"next_onset_{e}_s"] = nxt
        out[f"{s}_now"] = g[s].to_numpy()[i]
    return pd.DataFrame(out)


def split():
    """Seeded 70/15/15 subject split over the whole cohort and round-robin folds over train subjects; written once."""
    p = np.random.default_rng(CFG["seed"]).permutation(np.sort(pd.read_csv(RAW / "cohort.csv").subjectid.unique()))
    a, b = (np.cumsum([CFG["split"]["train"], CFG["split"]["val"]]) * len(p)).round().astype(int)
    splits = {k: sorted(v.tolist()) for k, v in zip(["train", "val", "test"], np.split(p, [a, b]))}
    fold = dict(zip(p[:a].tolist(), np.arange(a) % CFG["split"]["n_folds"]))
    folds = {str(s): int(fold[s]) for s in sorted(fold)}
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, obj in [("splits", splits), ("folds", folds)]:
        f, txt = PROCESSED / f"{name}.json", json.dumps(obj)
        if f.exists() and f.read_text() != txt:
            raise RuntimeError(f"{f} differs from the seeded split; never re-split silently")
        f.write_text(txt)
    return splits


def case(cs):
    caseid, subjectid = cs
    g = clean_case(pd.read_parquet(RAW / f"{caseid:04d}.parquet"))
    g.to_parquet(PROCESSED / "grid" / f"{caseid:04d}.parquet", index=False)
    ev = events(g)
    ev.insert(0, "caseid", caseid)
    return ev, prediction_table(g, caseid, subjectid)


def build():
    cohort = pd.read_csv(RAW / "cohort.csv")
    todo = [(c, s) for c, s in cohort[["caseid", "subjectid"]].itertuples(index=False) if (RAW / f"{c:04d}.parquet").exists()]
    splits = split()
    (PROCESSED / "grid").mkdir(exist_ok=True)
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(case, todo, chunksize=8))
    ev = pd.concat([r[0] for r in res], ignore_index=True)
    pt = pd.concat([r[1] for r in res], ignore_index=True)
    pt["split"] = pt.subjectid.map({s: k for k, v in splits.items() for s in v})
    ev.to_parquet(PROCESSED / "events.parquet", index=False)
    pt.to_parquet(PROCESSED / "pred_table.parquet", index=False)
    summary(ev, pt)


def summary(ev, pt):
    cases = pd.read_csv(RAW / "cases.csv").set_index("caseid")
    print(f"{pt.caseid.nunique()} cases, {len(pt)} prediction rows, split rows {pt.split.value_counts().to_dict()}")
    for e in EV:
        x, el = ev[ev.event == e], pt[pt[f"eligible_{e}"]]
        print(f"{e}: {len(x)} events in {x.caseid.nunique()} cases; {len(el)} eligible rows ({len(el) / len(pt):.1%}); "
              f"{pt[f'flag_{e}'].sum()} rows flagged ({el[f'flag_{e}'].sum()} of them eligible)")
        for h in W["horizons_min"]:
            y = el[f"y_{e}_{h}"]
            print(f"  {h:>2} min: {int(y.sum())} positive of {y.notna().sum()} labelled eligible ({y.mean():.2%}), {y.isna().sum()} censored")
        o = x.onset_sec.to_numpy()
        a, s, f, z = cases.loc[x.caseid, ["anestart", "opstart", "opend", "aneend"]].to_numpy().T
        phase = np.select([(o >= a) & (o < s), (o >= s) & (o < f), (o >= f) & (o <= z)],
            ["induction", "maintenance", "emergence"], "outside anestart..aneend")
        print("  onsets by phase:", pd.Series(phase).value_counts(normalize=True).round(3).to_dict())


def plot_events(n):
    """n random events (half per type, fixed seed), +-15 min around onset, from the saved outputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev, pt = pd.read_parquet(PROCESSED / "events.parquet"), pd.read_parquet(PROCESSED / "pred_table.parquet")
    cases = pd.read_csv(RAW / "cases.csv").set_index("caseid")
    rng = np.random.default_rng(CFG["seed"])
    pick = pd.concat([x.iloc[rng.choice(len(x), min(n // len(EV), len(x)), replace=False)] for _, x in ev.groupby("event")])
    out = REPORTS / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    ylim = {"map": (0, 210), "spo2": (45, 101)}
    for r in pick.itertuples():
        s, thr = EV[r.event]["signal"], EV[r.event]["threshold"]
        raw = pd.read_parquet(RAW / f"{r.caseid:04d}.parquet")
        g = pd.read_parquet(PROCESSED / "grid" / f"{r.caseid:04d}.parquet")
        w = ((g.t_sec - r.onset_sec).abs() <= 900).to_numpy()
        m = (g.t_sec[w] - r.onset_sec) / 60
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[3, 1])
        ax[0].plot(m, raw[s][w].clip(*ylim[s]), color="0.65", lw=1, label="raw (clipped to axis)")
        ax[0].plot(m, g[s][w], color="#2a78d6", lw=1.2, label="cleaned")
        a = w & g[f"art_{s}"].to_numpy()
        ax[0].plot((g.t_sec[a] - r.onset_sec) / 60, g[f"{s}_raw"][a].clip(*ylim[s]), "x", color="#d62728", ms=4, label="artifact")
        ax[0].axhline(thr, color="k", ls="--", lw=0.8)
        for o in ev[(ev.caseid == r.caseid) & (ev.event == r.event)].itertuples():
            ax[0].axvspan((o.onset_sec - r.onset_sec) / 60, (o.end_sec + STEP - r.onset_sec) / 60, color="#d62728", alpha=0.12)
        p = pt.t_sec[(pt.caseid == r.caseid) & pt[f"eligible_{r.event}"] & (pt.t_sec < r.onset_sec)]
        if len(p):
            t = (p.iat[-1] - r.onset_sec) / 60
            ax[0].axvspan(t - 10, t, ymax=0.04, color="#2a78d6", alpha=0.5, label=f"input window, last eligible t = {t:+.1f} min")
            ax[0].axvspan(t, t + 10, ymax=0.04, color="#eb6834", alpha=0.5, label="10 min label window")
        if s == "map":
            ax[1].plot(m, (raw.sbp - raw.dbp)[w], color="0.3", lw=0.8)
            ax[1].axhline(CFG["artifact"]["pulse_pressure_min"], color="k", ls="--", lw=0.8)
            ax[1].set_ylabel("monitor SBP-DBP")
        else:
            ax[1].plot(m, raw.pleth_amp[w], color="0.3", lw=0.8)
            ax[1].set_ylabel("pleth amplitude")
        for k in ("anestart", "opstart", "opend", "aneend"):
            x = (cases.at[r.caseid, k] - r.onset_sec) / 60
            if abs(x) <= 15:
                for b in ax:
                    b.axvline(x, color="0.4", ls=":", lw=0.8)
                ax[0].text(x, 1.01, k, transform=ax[0].get_xaxis_transform(), ha="center", fontsize=8)
        ax[0].set_ylim(*ylim[s])
        ax[0].set_xlim(-15, 15)
        ax[0].set_ylabel(s)
        ax[0].legend(loc="lower left", fontsize=7, frameon=False, ncol=3)
        ax[1].set_xlabel("minutes from onset")
        fig.suptitle(f"case {r.caseid} {r.event}: onset {r.onset_sec / 60:.1f} min, {r.duration_s} s, artifact {r.artifact_frac:.0%}", y=0.995)
        fig.tight_layout()
        fig.savefig(out / f"event_{r.caseid}_{r.event}_{r.onset_sec}.png", dpi=90)
        plt.close(fig)
    print(f"{len(pick)} plots -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-events", type=int, metavar="N", help="plot N random events from the saved outputs")
    args = ap.parse_args()
    plot_events(args.plot_events) if args.plot_events else build()
