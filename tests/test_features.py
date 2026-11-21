import numpy as np
import pandas as pd
import pytest

import orew.features as F
from orew.clean import STEP

N = 2 * 3600 // STEP  # 2 h case
T = np.arange(600, N * STEP, 30)
rng = np.random.default_rng(0)
SIGNALS = sorted({s for s, _, _ in F.SPEC})


def grid():
    g = pd.DataFrame(rng.normal(50, 10, (N, len(SIGNALS))).cumsum(0) / 10 + 60, columns=SIGNALS)
    g.loc[rng.random(N) < 0.1, ["map", "sbp", "spo2", "etco2"]] = np.nan
    g.loc[1000:1300, "map"] = np.nan  # 10 min gap
    for s in ("map", "spo2"):
        g[f"{s}_missing"] = g[s].isna().astype("int8")
        g[f"art_{s}"] = rng.random(N) < 0.05
    g.insert(0, "t_sec", np.arange(N) * STEP)
    return g


def clin(**markers):
    c = pd.DataFrame({"caseid": [1], "anestart": [-300.0], "opstart": [1500.0], "opend": [5000.0], "aneend": [6000.0],
                      "age": [60.0], "sex": ["M"], "height": [170.0], "weight": [70.0], "bmi": [24.2], "asa": [2.0],
                      "emop": [0], "department": ["Urology"], "optype": ["x"], "preop_ecg": ["NSR"],
                      "intraop_ebl": [500.0], **{p: [1.0] for p in F.PREOP}})
    for m, v in markers.items():
        c[m] = float(v)
    return F.clinical(c).iloc[0]


def test_columns_and_drug_subset():
    f = F.case_features(grid(), T, clin())
    assert list(f.columns) == F.FEATURES and len(f) == len(T)
    assert set(F.DRUG_COLS) <= set(F.FEATURES)
    assert sorted(F.DRUG_COLS) == sorted([*F.RATES, *(f"{r}_delta_5m" for r in F.RATES), *F.VOLATILE])
    assert f.sex.iat[0] == 1 and f.phase_maintenance.sum() > 0


def test_no_intraop():  # leakage #4
    assert not [c for c in F.FEATURES if "intraop" in c]
    assert not [c for c in clin().index if "intraop" in c]


@pytest.mark.parametrize("cut", [600, 1234, 2400, 4000, 6000])
def test_truncation(cut):  # leakage #2: deleting grid rows after cut leaves every feature at t <= cut unchanged
    g, c = grid(), clin()
    t = T[T <= cut]
    full = F.case_features(g, t, c)
    pd.testing.assert_frame_equal(F.case_features(g[g.t_sec <= cut].copy(), t, c), full)
    # and deleting the future still leaves features unchanged when values there are scrambled instead
    g2 = g.copy()
    g2.loc[g2.t_sec > cut, SIGNALS] = 999.0
    pd.testing.assert_frame_equal(F.case_features(g2, t, c), full)


PHASE = ["phase", *(f"phase_{p}" for p in F.PHASES), "min_since_phase_start", "min_since_anestart", "baseline_map"]


@pytest.mark.parametrize("t", [0, 200, 400, 900, 1500, 3000, 5000, 5500, 6000, 7000])
def test_phase_ignores_future_markers(t):  # leakage #5
    g, c = grid(), clin()
    base = F.case_features(g, [t], c)[PHASE]
    future = [m for m in F.MARKERS if c[m] > t]
    for change in (np.nan, t + 1, t + 10_000):
        c2 = clin(**{m: change for m in future})
        pd.testing.assert_frame_equal(F.case_features(g, [t], c2)[PHASE], base)


def test_phase_values():
    f = F.case_features(grid(), [0, 1000, 1500, 5100, 6600], clin()).set_index(pd.Index([0, 1000, 1500, 5100, 6600]))
    assert f.phase.tolist() == [1, 1, 2, 3, 4]
    assert f.min_since_phase_start.tolist() == pytest.approx([5, 1300 / 60, 0, 100 / 60, 10])
    assert f.min_since_anestart.loc[0] == pytest.approx(5)
    assert np.isnan(f.baseline_map.loc[0])  # window (0, 600] after max(anestart, 0) not over yet
    assert np.isfinite(f.baseline_map.loc[1000])  # window (0, 600] ended


def test_slope_exact():
    g = grid()
    g["map"] = 80 + 0.5 * g.t_sec / 60  # +0.5 mmHg per minute
    g.loc[rng.random(N) < 0.3, "map"] = np.nan  # gaps must not bias a least-squares slope
    f = F.case_features(g, T, clin())
    for m in (1, 3, 5, 10):
        assert np.allclose(f[f"map_slope_{m}m"].dropna(), 0.5, rtol=1e-5)
    assert f["map_slope_10m"].notna().all()
