"""Phase 2: cleaned grids, events, prediction table, subject splits (uv run python -m orew.labels [--plot-events 20])."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from orew import CFG, PROCESSED, RAW, REPORTS
from orew.clean import STEP, clean_case

EV = {k: v for k, v in CFG["events"].items() if isinstance(v, dict)}
W = CFG["windows"]


def events(g):
    """One row per event. A run is a maximal stretch of consecutive grid steps whose {signal}_raw (ffilled <= 30 s,
    artifact kept at its raw value) is below threshold; NaN ends a run. onset_sec / end_sec are the first / last grid
    time of the run, duration_s = steps * STEP (each step stands for its 2 s bin), artifact_frac = share of art_{signal}
    steps. A run is an event iff duration_s >= min_duration_s and artifact_frac <= max_artifact_frac."""
    t, out = g.t_sec.to_numpy(), []
    for e, c in EV.items():
        d = np.diff(np.r_[0, (g[f"{c['signal']}_raw"] < c["threshold"]).astype(int), 0])
        a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
        art = np.r_[0, np.cumsum(g[f"art_{c['signal']}"])]
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-events", type=int, metavar="N", help="plot N random events from the saved outputs")
    args = ap.parse_args()
    build()
