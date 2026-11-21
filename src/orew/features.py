"""Snapshot features at each prediction time t from the cleaned 2 s grid (rows t_sec <= t only) and pre-op clinical
fields (uv run python -m orew.features -> data/processed/features.parquet, one row per pred_table row)."""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from orew import PROCESSED, RAW
from orew.clean import STEP

ROWS = {m: m * 60 // STEP for m in (1, 3, 5, 10)}  # trailing window (t - m min, t] in grid rows
PAD = ROWS[10]
LEVEL = [("now", 0)] + [(s, m) for s in ("mean", "min", "slope") for m in (1, 3, 5, 10)]
BEAT5 = ["beat_pp", "beat_dpdt", "beat_sys_area", "beat_ppv", "beat_sbp_sd"]
RATES = ["ppf_rate", "rftn_ug_h", "phen_rate", "nepi_rate"]
VOLATILE = ["sevo", "des", "mac"]
SPEC = (  # (signal, stat, minutes); "now" = grid value at t, "delta" = now minus 5 min ago
    [(s, st, m) for s in ("map", "sbp", "dbp") for st, m in LEVEL]
    + [("hr", "now", 0), ("hr", "mean", 5), ("hr", "slope", 1), ("hr", "slope", 5)]
    + [(s, st, m) for s in BEAT5 for st, m in [("now", 0), ("mean", 1), ("slope", 5)]]
    + [(s, st, m) for s in RATES for st, m in [("now", 0), ("delta", 5)]]
    + [(s, "now", 0) for s in VOLATILE]
    + [("bis", "now", 0), ("bis", "slope", 5), ("bt", "now", 0), ("spo2", "now", 0)]
    + [("spo2", st, m) for st in ("min", "slope") for m in (1, 3, 5)]
    + [("etco2", "now", 0), ("etco2", "delta", 5)]
    + [(s, "now", 0) for s in ("rr", "tv", "peep", "pip", "fio2", "pleth_amp", "pleth_cv")]
)
MISS = ["map_missing", "spo2_missing", "art_map", "art_spo2"]  # fraction over the 10 min input window
MARKERS = ["anestart", "opstart", "opend", "aneend"]
PHASES = ["pre", "induction", "maintenance", "emergence", "post"]
PREOP = ["preop_htn", "preop_dm", "preop_hb", "preop_plt", "preop_pt", "preop_aptt", "preop_na", "preop_k",
         "preop_gluc", "preop_alb", "preop_ast", "preop_alt", "preop_bun", "preop_cr", "preop_ph", "preop_hco3",
         "preop_be", "preop_pao2", "preop_paco2", "preop_sao2"]  # every numeric preop_* in cases.csv (checked in clinical)
CLIN = ["age", "sex", "height", "weight", "bmi", "asa", "emop", "department"] + PREOP


def name(s, st, m):
    return s if st == "now" else f"{s}_{st}_{m}m"


FEATURES = ([name(*x) for x in SPEC] + ["shock_index", "pleth_amp_rel"] + [f"{c}_frac" for c in MISS]
            + ["phase"] + [f"phase_{p}" for p in PHASES] + ["min_since_phase_start", "min_since_anestart", "baseline_map"]
            + CLIN)
DRUG_COLS = [f for f in FEATURES if f.startswith(tuple(RATES + VOLATILE))]


def stat(w, st):
    """NaN-aware stat over each row of w (n_t x window, oldest first); slope by least squares in units per minute,
    NaN unless at least half the window is valid."""
    v = np.isfinite(w)
    y = np.where(v, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = v.sum(1)
        if st == "mean":
            return y.sum(1) / n
        if st == "min":
            return np.where(n > 0, np.where(v, w, np.inf).min(1), np.nan)
        x = np.arange(1 - w.shape[1], 1) * STEP / 60
        d = np.where(v, x - (v * x).sum(1, keepdims=True) / n[:, None], 0.0)
        # a 2-point slope across a gap edge reaches ~800 mmHg/min, so need half the window valid
        return np.where(n >= w.shape[1] / 2, (d * y).sum(1) / (d * d).sum(1), np.nan)


def clinical(cases):
    """Pre-op features (never intraop_*) and phase markers per caseid; department coded by its sorted unique values."""
    pre = [c for c in cases if c.startswith("preop_") and pd.api.types.is_numeric_dtype(cases[c])]
    assert pre == PREOP, f"numeric preop_* columns changed: {pre}"
    out = cases.set_index("caseid")[MARKERS + CLIN].copy()
    out["sex"] = out.sex.map({"M": 1, "F": 0})
    out["department"] = out.department.map({d: i for i, d in enumerate(sorted(cases.department.dropna().unique()))})
    return out.astype(float)


def case_features(g, t, c):
    """Features at times t (s) from grid g (t_sec = 0, STEP, ...) and clinical row c; uses only grid rows <= t
    and only phase markers <= t."""
    assert g.t_sec.iat[-1] == (len(g) - 1) * STEP, "grid must be gap-free from t_sec = 0"
    t = np.asarray(t)
    i = t // STEP + PAD  # row of t in the NaN-padded arrays
    win = i[:, None] + np.arange(1 - PAD, 1)  # trailing 10 min (t - 600, t]
    cache, out = {}, {}

    def col(s):
        if s not in cache:
            cache[s] = np.r_[np.full(PAD, np.nan), g[s].to_numpy(float)]
        return cache[s]

    for s, st, m in SPEC:
        a = col(s)
        out[name(s, st, m)] = a[i] if st == "now" else a[i] - a[i - ROWS[m]] if st == "delta" else stat(a[win[:, -ROWS[m]:]], st)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["shock_index"] = out["hr"] / out["sbp"]
        out["pleth_amp_rel"] = out["pleth_amp"] / stat(col("pleth_amp")[win], "mean")
    for s in MISS:
        out[f"{s}_frac"] = stat(col(s)[win], "mean")

    mk = c[MARKERS].to_numpy(float)
    phase = np.zeros(len(t), int)
    for k in range(4):  # latest marker already passed (NaN never passes); a future marker is never read
        phase = np.where(mk[k] <= t, k + 1, phase)
    out["phase"] = phase
    for k, p in enumerate(PHASES):
        out[f"phase_{p}"] = phase == k
    out["min_since_phase_start"] = (t - np.r_[0.0, mk][phase]) / 60  # pre phase counts from case start
    out["min_since_anestart"] = np.where(t >= mk[0], (t - mk[0]) / 60, np.nan)
    b = max(mk[0], 0) if np.isfinite(mk[0]) else np.nan
    first = g["map"][(g.t_sec > b) & (g.t_sec <= b + 600)].mean()
    out["baseline_map"] = np.where(t >= b + 600, first, np.nan)  # the window has ended, so only rows <= t
    for k in CLIN:
        out[k] = np.full(len(t), c[k])
    return pd.DataFrame(out)[FEATURES].astype("float32")


def _case(job):
    caseid, t, c = job
    f = case_features(pd.read_parquet(PROCESSED / "grid" / f"{caseid:04d}.parquet"), t, c)
    f.insert(0, "caseid", caseid)
    f.insert(1, "t_sec", t)
    return f


def build():
    t0 = time.time()
    pt = pd.read_parquet(PROCESSED / "pred_table.parquet", columns=["caseid", "t_sec"])
    clin = clinical(pd.read_csv(RAW / "cases.csv", encoding="utf-8-sig"))
    jobs = [(cid, x.t_sec.to_numpy(), clin.loc[cid]) for cid, x in pt.groupby("caseid", sort=False)]
    with ProcessPoolExecutor() as ex:
        f = pd.concat(ex.map(_case, jobs, chunksize=8), ignore_index=True)
    f = pt.merge(f, on=["caseid", "t_sec"], how="left", validate="one_to_one")  # pred_table row order
    f.to_parquet(PROCESSED / "features.parquet", index=False)
    print(f"features.parquet: {f.shape[0]} rows x {len(FEATURES)} features ({len(DRUG_COLS)} drug) in {time.time() - t0:.0f} s")
    print("NaN share, top 10:", f[FEATURES].isna().mean().sort_values(ascending=False).head(10).round(3).to_dict())


if __name__ == "__main__":
    build()
