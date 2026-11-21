"""Phase 1: cohort selection and per-case cache (uv run python -m orew.download [--n 5] [--check])."""
import argparse
import os
import resource
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import vitaldb
from vitaldb.utils import FMT_NAN

from orew import CFG, RAW, REPORTS
from orew.clean import FS, STEP, beat_grid, decimate, pleth_grid, bin_last

API = "https://api.vitaldb.net"
WORKERS = 6
NUMERIC = {k: [v] if isinstance(v, str) else v for k, v in CFG["tracks"]["numeric"].items()}
WAVE = CFG["tracks"]["wave"]
TRACKS = [n for v in NUMERIC.values() for n in v] + list(WAVE.values())
socket.setdefaulttimeout(60)  # a stalled read must raise so attempt() retries; workers inherit it via import


def cohort():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in ("cases", "trks"):
        if not (RAW / f"{name}.csv").exists():
            pd.read_csv(f"{API}/{name}", encoding="utf-8-sig").to_csv(RAW / f"{name}.csv", index=False)
    if (RAW / "cohort.csv").exists():
        return pd.read_csv(RAW / "cohort.csv")
    cases = pd.read_csv(RAW / "cases.csv")
    trks = pd.read_csv(RAW / "trks.csv")
    c = CFG["cohort"]
    has = trks[trks.tname.isin(c["required_tracks"])].groupby("caseid").tname.nunique()
    ok = (
        (pd.to_numeric(cases.age, errors="coerce") >= c["min_age"])
        & (cases.ane_type == c["ane_type"])
        & (cases.caseend - cases.casestart >= c["min_duration_s"])
        & cases.caseid.isin(has[has == len(c["required_tracks"])].index)
    )
    elig = cases[ok].sort_values("caseid")
    pick = np.random.default_rng(CFG["seed"]).choice(len(elig), CFG["n_cases"], replace=False)
    out = elig.iloc[pick][["caseid", "subjectid"]]
    out.to_csv(RAW / "cohort.csv", index=False)
    print(f"{len(elig)} eligible cases, sampled {len(out)}")
    return out


def wave(vf, name, nsamp):
    """Native-rate wave, sample i at i/FS s after file start, NaN where missing."""
    x = np.full(nsamp, np.nan, dtype=np.float32)
    trk = vf.trks.get(name)
    if trk is None:
        return x
    if trk.srate != FS:
        raise ValueError(f"{name} srate {trk.srate}")
    gain, offset = (trk.gain, trk.offset) if trk.fmt > 2 else (1.0, 0.0)
    for r in trk.recs:
        s = int(np.ceil((r["dt"] - vf.dtstart) * FS))  # same placement as vitaldb get_track_samples
        e = min(s + len(r["val"]), nsamp)
        # packed files fill gaps with a per-format sentinel that vitaldb's reader does not mask
        x[s:e] = np.where(r["val"][: e - s] == FMT_NAN[trk.fmt], np.nan, r["val"][: e - s] * gain + offset)
    return x


def process(caseid):
    path = RAW / f"{caseid:04d}.vital"
    try:
        urlretrieve(f"{API}/{vitaldb.dataset.DATASET_VERSION}/{caseid}.vital", path)  # raises if truncated
        vf = vitaldb.VitalFile()  # VitalFile(path) ignores load_vital's False on a corrupt file
        if not vf.load_vital(str(path), TRACKS):
            raise OSError("vitaldb could not read the .vital")
    finally:
        path.unlink(missing_ok=True)
    missing = [t for t in CFG["cohort"]["required_tracks"] if t not in vf.trks]
    if missing:
        raise ValueError(f"missing {missing}")
    dur = vf.dtend - vf.dtstart
    n, nsamp = int(np.ceil(dur / STEP)) + 1, int(np.ceil(dur * FS))
    num = {"t_sec": np.arange(n) * STEP}
    for key, names in NUMERIC.items():
        col = np.full(n, np.nan)
        for name in names:  # aliases: first non-missing in list order, per grid time
            if name in vf.trks:
                recs = vf.trks[name].recs
                col = np.where(np.isnan(col), bin_last([r["dt"] - vf.dtstart for r in recs], [r["val"] for r in recs], n), col)
        num[key] = col
    art = wave(vf, WAVE["art"], nsamp)
    df = pd.concat([pd.DataFrame(num), beat_grid(art, n), pleth_grid(wave(vf, WAVE["pleth"], nsamp), n)], axis=1)
    np.save(RAW / f"{caseid:04d}_art100.npy", decimate(art))
    tmp = RAW / f"{caseid:04d}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, RAW / f"{caseid:04d}.parquet")  # parquet present = case done


def attempt(caseid):
    t0, err = time.time(), None
    for k in range(3):
        try:
            process(caseid)
            err = None
            break
        except Exception as e:
            err = repr(e)
            if k < 2:
                time.sleep(10 * 3**k)  # 10 s then 30 s, to ride out a brief network drop or throttling
    return err, time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS


def run(ids):
    todo = [c for c in ids if not (RAW / f"{c:04d}.parquet").exists()]
    print(f"{len(ids) - len(todo)} cached, {len(todo)} to download")
    fails = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, (c, (err, sec, rss)) in enumerate(zip(todo, ex.map(attempt, todo)), 1):
            print(f"[{i}/{len(todo)}] {c}: {err or 'ok'} {sec:.0f}s, worker peak RSS {rss:.0f} MB", flush=True)
            if err:
                fails.append((c, err))
    f = RAW / "failed.csv"
    # ids in this run are now cached or re-logged below; earlier failures outside it still have no parquet
    old = pd.read_csv(f).query("caseid not in @ids") if f.exists() else None
    pd.concat([old, pd.DataFrame(fails, columns=["caseid", "error"])]).to_csv(f, index=False)
    print(f"{len(fails)} failed -> {f}")


def check(ids):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = pd.read_csv(RAW / "cases.csv").set_index("caseid")
    dfs = {c: pd.read_parquet(RAW / f"{c:04d}.parquet") for c in ids}
    print("% non-NaN on the 2 s grid (caseid columns)")
    print(pd.DataFrame({c: d.drop(columns="t_sec").notna().mean() * 100 for c, d in dfs.items()}).round(1).to_string())
    print("grid end vs cases.csv caseend (s):", {c: (int(d.t_sec.iloc[-1]), int(cases.caseend[c])) for c, d in dfs.items()})
    trks = pd.read_csv(RAW / "trks.csv")
    coh = trks[trks.caseid.isin(cohort().caseid)]
    print(f"cohort cases ({coh.caseid.nunique()}) having each SPEC track")
    print(coh.groupby("tname").caseid.nunique().reindex(TRACKS + ["SNUADC/ECG_II"], fill_value=0).to_string())
    out = REPORTS / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    for c, d in dfs.items():
        m = d.t_sec / 60
        fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        ax[0].plot(m, d["map"], lw=0.8, color="#2a78d6", label="ART MAP")
        ax[0].plot(m, d.nibp_map, "o", ms=4, color="#eb6834", label="NIBP MAP")
        ax[0].legend(loc="upper right", frameon=False)
        for a, col, unit in zip(ax, ["map", "hr", "spo2"], ["MAP (mmHg)", "HR (bpm)", "SpO2 (%)"]):
            if col != "map":
                a.plot(m, d[col], lw=0.8, color="#2a78d6")
            a.set_ylabel(unit)
            a.grid(alpha=0.2)
            for k in ("anestart", "opstart", "opend", "aneend"):
                a.axvline(cases.at[c, k] / 60, color="0.4", ls="--", lw=0.8)
        for k in ("anestart", "opstart", "opend", "aneend"):
            ax[0].text(cases.at[c, k] / 60, 1.02, k, transform=ax[0].get_xaxis_transform(), ha="center", fontsize=8)
        ax[2].set_xlabel("minutes from case start")
        fig.suptitle(f"case {c}", y=0.995)
        fig.tight_layout()
        fig.savefig(out / f"case_{c}.png", dpi=100)
        plt.close(fig)
    print(f"plots -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--check", action="store_true", help="availability, track counts and plots for the first n cases")
    args = ap.parse_args()
    ids = cohort().caseid.head(args.n).tolist()
    check(ids) if args.check else run(ids)
