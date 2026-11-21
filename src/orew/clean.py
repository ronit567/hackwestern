"""Causal 2 s grid features. Grid time g holds only data with timestamp in (g - STEP, g]."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from orew import CFG

STEP = CFG["grid"]["step_s"]
FS = CFG["grid"]["wave_hz_native"]
DEC = FS // CFG["grid"]["wave_hz_model"]
BIN = FS * STEP  # wave samples per grid bin

PROM = 10.0  # mmHg; candidate peaks need this prominence (rejects noise and quantisation steps)
WLEN = int(1.5 * FS) + 1  # odd: scipy searches prominence bases in [p - WLEN//2, p + WLEN//2]
# a beat peak is the highest candidate within +-REFR, which drops diastolic waves (~0.3 s after systole)
REFR = int(0.35 * FS)  # shortcut: caps detectable HR at ~170 bpm, make adaptive if tachycardia matters
K = 5  # dP/dt over 10 ms: ART is quantised at ~1 mmHg, so a 2 ms difference only resolves ~494 mmHg/s
BEAT_WIN = "12s"  # trailing window for breath-to-breath PP swing and SBP variability
PLETH_WIN = 30 // STEP  # trailing grid bins for pleth amplitude CV


def bin_last(t, v, n):
    """Last finite value with timestamp t in (g - STEP, g] for g = 0, STEP, ..., (n - 1) * STEP."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    ok = np.isfinite(v)
    o = np.argsort(t[ok], kind="stable")
    t, v = t[ok][o], v[ok][o]
    i = np.ceil(t / STEP).astype(int)
    last = np.r_[i[1:] != i[:-1], True] & (i < n)
    out = np.full(n, np.nan)
    out[i[last]] = v[last]
    return out


def to_grid(idx, df, n):
    """Mean of rows whose wave sample index idx falls in bin (g - STEP, g]; empty bins NaN."""
    return df.groupby(-(-idx // BIN)).mean().reindex(range(n))


def beats(x):
    """Per-beat ART features, beat = foot to next foot, stamped with available_at (sample index)."""
    q, d = find_peaks(x, plateau_size=1)
    # a local max is settled once its plateau has ended and its prominence window has been seen
    settled = np.maximum(q + WLEN // 2, d["right_edges"] + 1)  # non-decreasing in q
    c = q[peak_prominences(x, q, wlen=WLEN)[0] >= PROM]
    lo, hi = np.searchsorted(c, c - REFR), np.searchsorted(c, c + REFR, "right")
    p = c[[lo[i] + np.argmax(x[c[lo[i] : hi[i]]]) == i for i in range(len(c))]]  # ties: earliest wins
    # p is final once every local max up to p + REFR is settled
    known = np.maximum(p + REFR + 1, settled[np.searchsorted(q, p + REFR, "right") - 1])
    feet = np.array([a + np.argmin(x[a:b]) for a, b in zip(p[:-1], p[1:])], dtype=int)
    nan_cum = np.r_[0, np.cumsum(np.isnan(x))]
    rows = []
    for j in range(1, len(p) - 1):  # beat j: feet[j-1] -> feet[j], systolic peak p[j]
        f0, pk, f1 = feet[j - 1], p[j], feet[j]
        dur = f1 - f0
        if nan_cum[p[j + 1] + 1] - nan_cum[p[j - 1]] or not FS // 4 <= dur <= 2 * FS:
            continue  # gap inside the beat, or HR outside 30-240
        seg = x[f0 : f1 + 1]
        dbp = x[f0]
        rows.append((
            known[j + 1],  # feet[j] needs peak j+1
            x[pk] - dbp,
            (x[f0 + K : pk + 1] - x[f0 : pk + 1 - K]).max() * FS / K if pk - f0 >= K else np.nan,
            (seg[: dur // 3] - dbp).sum() / FS,  # shortcut: systole = first third of the beat, no notch detection
            60 * FS / dur,
            x[pk],
        ))
    return pd.DataFrame(rows, columns=["available_at", "pp", "dpdt", "sys_area", "hr", "sbp"], dtype=float)


def beat_grid(x, n):
    """Beat features on the grid; trailing-window stats only use beats available by then."""
    b = beats(x)
    r = b.set_index(pd.to_timedelta(b.available_at / FS, unit="s"))[["pp", "sbp"]].rolling(BEAT_WIN, min_periods=3)
    b["ppv"] = ((r.pp.max() - r.pp.min()) / r.pp.mean()).to_numpy()
    b["sbp_sd"] = r.sbp.std().to_numpy()
    g = to_grid(b.available_at.to_numpy(int), b.drop(columns=["available_at", "sbp"]), n)
    return g.add_prefix("beat_").reset_index(drop=True)


def pleth_grid(x, n):
    """Pleth peak-to-trough per bin (NaN if any sample missing) and its trailing 30 s CV."""
    y = np.full((n - 1) * BIN + 1, np.nan, dtype=np.float32)
    y[: min(len(x), len(y))] = x[: len(y)]
    blocks = y[1:].reshape(n - 1, BIN)  # bin k >= 1 holds samples (BIN*(k-1), BIN*k]
    amp = pd.Series(np.r_[np.nan, blocks.max(1) - blocks.min(1)], dtype=float)
    r = amp.rolling(PLETH_WIN, min_periods=PLETH_WIN // 2)
    return pd.DataFrame({"pleth_amp": amp, "pleth_cv": r.std() / r.mean()})


def decimate(x):
    """Causal block mean to the model rate; output sample i covers [i/100, (i+1)/100) s."""
    m = len(x) // DEC
    return x[: m * DEC].reshape(m, DEC).mean(1).astype(np.float32)
