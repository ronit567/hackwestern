import numpy as np
from scipy.signal import find_peaks

from orew.clean import FS, STEP, beat_grid, beats, bin_last, decimate, pleth_grid

DUR = 300
TS = [37.0, 101.5, 150.0, 222.2, 299.0]
rng = np.random.default_rng(0)
t = np.arange(DUR * FS) / FS


def art_wave():
    """ART-like wave: harmonics, drifting HR, respiratory PP swing, pressure drop, noise, a gap and clipping."""
    ph = 2 * np.pi * np.cumsum(1.25 + 0.1 * np.sin(2 * np.pi * t / 120)) / FS
    pp = 20 * (1 + 0.15 * np.sin(2 * np.pi * 0.25 * t))
    mean = 90 - 30 / (1 + np.exp(-(t - 150) / 10))
    x = mean + pp * (np.sin(ph) + 0.3 * np.sin(2 * ph + 1) + 0.1 * np.sin(3 * ph + 2)) + rng.normal(0, 0.5, t.size)
    x[100 * FS : 103 * FS] = np.nan
    x[200 * FS : 202 * FS] = np.minimum(x[200 * FS : 202 * FS], 85)  # flat-topped beats -> plateaus
    return x.astype(np.float32)


def cut(x, T):
    return x[: int(T * FS) + 1]  # samples with timestamp <= T


def n_grid(T):
    return int(T // STEP) + 1


def test_beat_grid_truncation():
    x = art_wave()
    full, full_beats = beat_grid(x, n_grid(DUR)).to_numpy(), beats(x)
    # also cut just after peaks, where a hidden look-ahead would show up
    peaks = find_peaks(x, prominence=10)[0][::15]
    for T in TS + [(p + d) / FS for p in peaks for d in (1, 50, 200, 330, 374, 375, 376)]:
        xt = cut(x, T)
        np.testing.assert_array_equal(full[: n_grid(T)], beat_grid(xt, n_grid(T)).to_numpy())
        b = beats(xt)
        np.testing.assert_array_equal(
            full_beats[full_beats.available_at < len(xt)].to_numpy(), b[b.available_at < len(xt)].to_numpy()
        )


def test_pleth_and_decimate_truncation():
    x = art_wave() / 2
    full, dec = pleth_grid(x, n_grid(DUR)).to_numpy(), decimate(x)
    for T in TS:
        xt = cut(x, T)
        np.testing.assert_array_equal(full[: n_grid(T)], pleth_grid(xt, n_grid(T)).to_numpy())
        d = decimate(xt)
        np.testing.assert_array_equal(dec[: len(d)], d)


def test_bin_last_truncation_and_causality():
    on_grid = rng.choice(n_grid(DUR), 20, replace=False) * STEP  # records exactly at grid times
    ts = np.unique(np.r_[rng.uniform(0, DUR, 380), on_grid])
    v = rng.normal(80, 10, ts.size)
    v[::17] = np.nan
    shuf = rng.permutation(ts.size)
    full = bin_last(ts[shuf], v[shuf], n_grid(DUR))
    for T in TS:
        keep = ts <= T
        np.testing.assert_array_equal(full[: n_grid(T)], bin_last(ts[keep], v[keep], n_grid(T)))
    for k, g in enumerate(np.arange(n_grid(DUR)) * STEP):
        inbin = (ts > g - STEP) & (ts <= g) & np.isfinite(v)
        np.testing.assert_equal(full[k], v[inbin][-1] if inbin.any() else np.nan)


def test_beat_sanity():
    x = (80 + 20 * np.sin(2 * np.pi * 1.2 * t) + rng.normal(0, 0.3, t.size)).astype(np.float32)
    g = beat_grid(x, n_grid(DUR))
    assert abs(g.beat_pp.median() - 40) < 2
    assert abs(g.beat_hr.median() - 72) < 1
    assert g.beat_pp.notna().mean() > 0.95
