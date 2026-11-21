import numpy as np
import pandas as pd

from orew import CFG
from orew.clean import BEAT, FILL, STEP, clean_case

COLS = list(CFG["tracks"]["numeric"]) + BEAT + ["pleth_amp", "pleth_cv"]
N = 3 * 3600 // STEP  # 3 h case
rng = np.random.default_rng(0)


def steps(a_min, b_min):
    """Grid indices for [a, b) minutes."""
    return slice(int(a_min * 60) // STEP, int(b_min * 60) // STEP)


def raw_case():
    """Synthetic raw grid: MAP ~80 and SpO2 ~98 with gaps, flush spikes, zeroed line, NIBP re-sends, probe-off."""
    d = pd.DataFrame(rng.normal(50, 10, (N, len(COLS))), columns=COLS)
    d.insert(0, "t_sec", np.arange(N) * STEP)
    d["map"] = 80 + rng.normal(0, 3, N)
    d["sbp"], d["dbp"] = d["map"] + 35, d["map"] - 20
    d["spo2"] = np.round(98 + rng.normal(0, 0.5, N))
    d["pleth_amp"] = 20 + rng.normal(0, 2, N)
    d["nibp_map"] = np.repeat(rng.normal(80, 8, N // 150 + 1), 150)[:N]  # cuff value re-sent every 2 s
    d.loc[rng.random(N) < 0.3, ["fio2", "tv", "peep"]] = np.nan  # ventilator ~7 s cadence
    d.loc[d.index[:N // 2], "rftn50_rate"] = np.nan
    d.loc[d.index[N // 2 :], "rftn_rate"] = np.nan
    for a, b in [(5, 5.3), (20, 21.5), (70, 70.2)]:  # short, long and 12 s gaps
        d.loc[d.index[steps(a, b)], ["map", "sbp", "dbp", "spo2"]] = np.nan
    d.loc[d.index[steps(0, 4)], ["map", "sbp", "dbp"]] = 0.0  # line not yet placed
    for a in (30, 45.05, 100):  # flush: ramp, plateau above 200, ramp down, flat wave
        i = steps(a, a).start
        d.loc[i : i + 6, "map"] = [130, 230, 250, 250, 170, 140, 90]
        d.loc[i : i + 5, "sbp"] = d.loc[i : i + 5, "map"] + 3
        d.loc[i : i + 5, "dbp"] = d.loc[i : i + 5, "map"] - 3
    d.loc[d.index[steps(60, 61)], "map"] = -495.0  # raw ADC 0
    d.loc[d.index[steps(130, 132)], "pleth_amp"] = np.nan  # probe off: pleth lost, SpO2 reads low
    d.loc[d.index[steps(130.2, 132)], "spo2"] = 80.0
    return d


def test_clean_case_truncation():
    raw = raw_case()
    full = clean_case(raw)
    for T in [0, 2, 598, 1802, 2700, 2710, 6000, 7202, 7840, 7900, (N - 1) * STEP]:
        k = T // STEP + 1
        pd.testing.assert_frame_equal(full.iloc[:k], clean_case(raw.iloc[:k]))


def test_clean_case_rules():
    raw = raw_case()
    g = clean_case(raw)
    assert g.art_map[steps(0, 4)].all() and g["map"][steps(0, 4)].isna().all()  # zeroed line
    i = steps(30, 30).start
    assert g.art_map[i : i + 6].all() and not g.art_map[i + 6]  # whole flush flagged, recovery sample kept
    assert (g["map"][i : i + 6] == raw["map"][i - 1]).all()  # bridged by the last valid MAP
    assert (g[BEAT][i : i + 6 + 6] == raw[BEAT].iloc[i - 1]).all().all()  # artifact beats replaced, 12 s window
    assert g.art_spo2[steps(130.2, 132)].all() and not g.art_spo2[steps(120, 130)].any()
    # ffill: the first FILL steps of a gap are filled, the rest stay NaN and are masked
    gap = steps(20, 21.5)
    assert g["map"][gap][:FILL].notna().all() and g["map"][gap][FILL:].isna().all()
    assert (g.map_missing[gap][FILL:] == 1).all() and g.map_missing[steps(25, 29)].eq(0).all()
    assert g.fio2.notna().all() and g.fio2_missing.eq(0).all()  # ventilator cadence bridged
    assert g.nibp_map.notna().sum() == raw.nibp_map.ne(raw.nibp_map.shift()).sum()
    np.testing.assert_allclose(g.rftn_ug_h, np.r_[20 * raw.rftn_rate[: N // 2], 50 * raw.rftn50_rate[N // 2 :]])
    assert "rftn_rate" not in g and "rftn50_rate" not in g
