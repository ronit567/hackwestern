import json

import numpy as np
import pandas as pd
import pytest

import orew.labels as L
from orew import CFG, PROCESSED
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
    d["pleth_amp"], d["pleth_cv"] = 20 + rng.normal(0, 2, N), 0.1
    d["nibp_map"] = np.repeat(rng.normal(80, 8, N // 150 + 1), 150)[:N]  # cuff value re-sent every 2 s
    d.loc[rng.random(N) < 0.3, ["fio2", "tv", "peep"]] = np.nan  # ventilator ~7 s cadence
    d.loc[d.index[:N // 2], "rftn50_rate"] = np.nan
    d.loc[d.index[N // 2 :], "rftn_rate"] = np.nan
    for a, b in [(5, 5.3), (20, 21.5), (70, 70.2)]:  # short, long and 12 s gaps
        d.loc[d.index[steps(a, b)], ["map", "sbp", "dbp", "spo2"]] = np.nan
    d.loc[d.index[steps(0, 4)], ["map", "sbp", "dbp"]] = 0.0  # line not yet placed
    d.loc[d.index[steps(4, 7)], "map"] = 55.0  # placed while MAP is low: event from the first real sample
    d.loc[d.index[steps(176.5, 178)], "map"] = 55.0  # 90 s low MAP, then the line is removed: event up to the last real sample
    d.loc[d.index[steps(178, 180)], ["map", "sbp", "dbp"]] = 0.0
    d.loc[d.index[steps(179.6, 180)], "spo2"] = 30.0  # probe taken off: SpO2 ends before the file too
    for a in (30, 45.05, 100):  # flush: ramp, plateau above 200, ramp down, flat wave
        i = steps(a, a).start
        d.loc[i : i + 6, "map"] = [130, 230, 250, 250, 170, 140, 90]
        d.loc[i : i + 5, "sbp"] = d.loc[i : i + 5, "map"] + 3
        d.loc[i : i + 5, "dbp"] = d.loc[i : i + 5, "map"] - 3
    d.loc[d.index[steps(60, 61)], "map"] = -495.0  # raw ADC 0
    i = steps(75, 75).start  # 50 s flush: 30 s above 200, then a 185 plateau sent without SBP/DBP
    d.loc[i : i + 24, "map"] = [130] + [230] * 15 + [185] * 9
    d.loc[i + 16 : i + 24, ["sbp", "dbp"]] = np.nan
    d.loc[d.index[steps(130, 134)], ["pleth_amp", "pleth_cv"]] = np.nan  # probe off: pleth lost, SpO2 low 3.8 min
    d.loc[d.index[steps(140, 144)], "pleth_cv"] = 0.9  # motion: pleth erratic, SpO2 steps down for 3.5 min
    d.loc[d.index[steps(140.5, 144)], "spo2"] = 86.0
    d.loc[d.index[steps(160, 165)], ["pleth_amp", "spo2"]] = [0.8, np.nan]  # 5 min probe off, back on a weak pleth
    d.loc[d.index[steps(165, 167.5)], ["pleth_amp", "spo2"]] = [4.0, 82.0]  # < 2.5 min: then it is the new level
    d.loc[d.index[steps(170, 174)], "pleth_amp"] = 95.0  # pleth pinned on the ADC rail, calm CV
    d.loc[d.index[steps(170.5, 174)], "spo2"] = 85.0
    d.loc[d.index[steps(130.2, 134)], "spo2"] = 80.0
    d.loc[d.index[steps(40, 43)], "map"] = 55.0  # real hypotension with three below-threshold artifact points
    d.loc[[steps(41, 41).start + k for k in (0, 10, 20)], "map"] = 5.0
    d.loc[d.index[steps(42, 42.2)], ["map", "sbp", "dbp"]] = np.nan  # a 12 s dropout inside it doesn't end the run
    d.loc[d.index[steps(50, 50.9)], "map"] = 60.0  # 54 s dip: no event
    d.loc[d.index[steps(65, 66)], "map"] = 60.0  # 60 s low MAP with a flush spike mid-way: still one event
    d.loc[steps(65.5, 65.5).start, "map"] = 150.0
    d.loc[d.index[steps(80, 81)], "map"] = 60.0  # exactly 30 steps = 60 s: event
    d.loc[d.index[steps(85, 85 + 58 / 60)], "map"] = 60.0  # 29 steps: no event
    d.loc[steps(90, 90).start : steps(92, 92).start, "map"] = 60.0  # 61 steps: ends at 5520 s, on a prediction time
    d.loc[d.index[steps(87, 88 + 1 / 3)], "map"] = 55.0  # 80 s run, 75 % zero line between real ends: no event
    d.loc[d.index[steps(87 + 1 / 6, 88 + 1 / 6)], ["map", "sbp", "dbp"]] = 0.0
    d.loc[d.index[steps(110, 113)], "spo2"] = 85.0  # real hypoxemia, pleth fine
    d.loc[d.index[steps(119, 124)], ["hr", "etco2"]] = np.nan  # off the monitors, erratic pleth, SpO2 falls by <= 1
    d.loc[d.index[steps(119.5, 124)], "pleth_cv"] = 0.9
    k = steps(120, 124)
    d.loc[d.index[k], "spo2"] = np.round(np.linspace(95, 80, k.stop - k.start))
    k = steps(174.5, 179.5)  # real gradual desaturation (<= 1 per reading) on a pleth with spikes past the rail
    d.loc[d.index[k], "spo2"] = np.round(np.linspace(97, 82, k.stop - k.start))
    d.loc[d.index[k][::8], "pleth_amp"] = 300.0
    d.loc[d.index[steps(146, 146 + 40 / 60)], "map"] = 60.0  # 40 s below threshold, then a 60 s dropout: no event
    d.loc[d.index[steps(146 + 40 / 60, 147 + 40 / 60)], ["map", "sbp", "dbp"]] = np.nan
    d.loc[d.index[steps(150, 158)], ["map", "sbp", "dbp"]] = np.nan  # 8 min without MAP
    d.loc[steps(158, 158).start, "map"] = 250.0  # and it resumes on a flush sample
    k = steps(95, 96.5)  # blood draw: MAP decays 80 -> 40 (< 40 per step), sent without SBP/DBP, no ART beats
    d.loc[d.index[k], "map"] = 40 + 40 * np.exp(-np.arange(k.stop - k.start) / 8)
    d.loc[d.index[k], ["sbp", "dbp"] + BEAT] = np.nan
    return d


def test_clean_case_truncation():
    raw = raw_case()
    full = clean_case(raw)
    x = full[[c for c in full if c.startswith("art_") or c.endswith("_missing")] + ["nibp_map"]].astype(float).fillna(-1)
    edge = np.flatnonzero((x != x.shift()).any(axis=1))  # every row where a flag, mask or cuff reading changes
    ks = {k + d for k in edge for d in (0, 1, 2)} | {0, 299, 901, 3600, N - 1}
    for k in sorted(k for k in ks if k < N):  # truncate right after grid row k
        pd.testing.assert_frame_equal(full.iloc[: k + 1], clean_case(raw.iloc[: k + 1]))


def test_clean_case_rules():
    raw = raw_case()
    g = clean_case(raw)
    assert g.art_map[steps(0, 4)].all() and g["map"][steps(0, 4)].isna().all()  # zeroed line
    i = steps(30, 30).start
    assert g.art_map[i : i + 6].all() and not g.art_map[i + 6]  # whole flush flagged, recovery sample kept
    assert (g["map"][i : i + 6] == raw["map"][i - 1]).all()  # bridged by the last valid MAP
    assert (g[BEAT][i : i + 6 + 6] == raw[BEAT].iloc[i - 1]).all().all()  # artifact beats replaced, 12 s window
    i = steps(75, 75).start
    assert g.art_map[i : i + 25].all() and not g.art_map[i + 25]  # plateau outlives the reference, still flagged
    assert g.art_map[steps(95, 96.5)].all()  # blood-draw decay: no SBP/DBP and no beat
    assert not g.art_map[steps(96.5, 99)].any()
    for a, b in [(130.2, 134), (140.5, 144), (165, 167.5), (170.5, 174)]:  # probe off, motion, weak after flat, rail
        assert g.art_spo2[steps(a, b)].all()
    assert not g.art_spo2[steps(100, 119)].any()  # includes the real hypoxemia at 110-113 min
    assert g.art_spo2[steps(120, 124)].all() and not g.art_spo2[steps(124, 130)].any()  # no HR, no ETCO2, poor pleth
    assert not g.art_spo2[steps(174.5, 179.5)].any()  # gradual fall: pleth spikes alone don't make it artifact
    # ffill: the first FILL steps of a gap are filled, the rest stay NaN and are masked
    gap = steps(20, 21.5)
    assert g["map"][gap][:FILL].notna().all() and g["map"][gap][FILL:].isna().all()
    assert (g.map_missing[gap][FILL:] == 1).all() and g.map_missing[steps(25, 29)].eq(0).all()
    assert g[BEAT][gap][FILL:].isna().all().all() and (g.beat_pp_missing[gap][FILL:] == 1).all()  # beats need MAP
    assert g.fio2.notna().all() and g.fio2_missing.eq(0).all()  # ventilator cadence bridged
    assert g.nibp_map.notna().sum() == raw.nibp_map.ne(raw.nibp_map.shift()).sum()
    np.testing.assert_allclose(g.rftn_ug_h, np.r_[20 * raw.rftn_rate[: N // 2], 50 * raw.rftn50_rate[N // 2 :]])
    assert "rftn_rate" not in g and "rftn50_rate" not in g
    raw.loc[raw.index[steps(119, 124)], "etco2"] = 30.0  # still ventilated: ECG off alone is not artifact
    assert not clean_case(raw).art_spo2[steps(120, 124)].any()


H = CFG["windows"]["horizons_min"]


def table(raw):
    g = clean_case(raw)
    return g, L.events(g), L.prediction_table(g, 7, 70)


def test_event_conventions_and_artifact_runs():
    _, ev, _ = table(raw_case())
    hypo = ev[ev.event == "hypotension"]
    # ADC -495 (60-61 min) is an all-artifact run; 54 s and 58 s dips are too short; the zero lines at 0-4 and 178-180
    # min are trimmed off the low MAP next to them (untrimmed they'd make those runs > 50 % artifact)
    assert hypo.onset_sec.tolist() == [4 * 60, 40 * 60, 65 * 60, 80 * 60, 90 * 60, 176.5 * 60]
    assert hypo.duration_s.tolist() == [180, 180, 60, 60, 122, 90] and (hypo.end_sec == hypo.onset_sec + hypo.duration_s - STEP).all()
    assert hypo.artifact_frac.tolist() == [0, 3 / 90, 1 / 30, 0, 0, 0]
    g = clean_case(raw_case())
    assert g["map"][steps(146, 146 + 40 / 60 + FILL * STEP / 60)].eq(60).all()  # the causal grid fills the dropout's head
    assert ev[ev.event == "hypoxemia"].onset_sec.tolist()[:1] == [110 * 60] and len(ev[ev.event == "hypoxemia"]) == 2  # probe-off at 130 min and motion at 140 min are not events


def test_nibp_never_changes_hypotension():
    raw = raw_case()
    _, ev, pt = table(raw)
    raw["nibp_map"] = rng.uniform(0, 300, N)
    raw.loc[rng.random(N) < 0.5, "nibp_map"] = np.nan
    _, ev2, pt2 = table(raw)
    hcols = [c for c in pt if "hypotension" in c or c == "map_now"]
    pd.testing.assert_frame_equal(ev[ev.event == "hypotension"], ev2[ev2.event == "hypotension"])
    pd.testing.assert_frame_equal(pt[hcols], pt2[hcols])


def test_every_time_kept_and_only_documented_exclusions():
    g, ev, pt = table(raw_case())
    last = g.t_sec.iat[-1]
    assert pt.t_sec.tolist() == list(range(600, last + 1, 30))  # every 30 s from input_s, none dropped
    assert len(pt) == len(pt.drop_duplicates("t_sec"))
    for e, c in L.EV.items():
        s, x = c["signal"], ev[ev.event == e]
        bad = (g[s].isna() | g[f"art_{s}"]).to_numpy()
        for r in pt.itertuples():
            in_progress = ((x.onset_sec <= r.t_sec) & (r.t_sec <= x.end_sec)).any()
            frac = bad[(g.t_sec > r.t_sec - 600) & (g.t_sec <= r.t_sec)].mean()
            assert getattr(r, f"eligible_{e}") == (not in_progress and frac <= 0.5)
            if in_progress:
                assert not getattr(r, f"eligible_{e}")
            assert getattr(r, f"flag_{e}") == ((r.t_sec > x.end_sec) & (r.t_sec <= x.end_sec + 300)).any()
    assert not pt.eligible_hypotension[(pt.t_sec >= 158 * 60) & (pt.t_sec <= 160 * 60)].any()  # MAP gap
    assert not pt.eligible_hypotension[(pt.t_sec >= 40 * 60) & (pt.t_sec < 43 * 60)].any()  # in progress
    r = pt.set_index("t_sec")  # t == end_sec: the run may go on, so still in progress and not yet "after" the event
    assert not r.eligible_hypotension[5520] and not r.flag_hypotension[5520] and r.flag_hypotension[5550]


def test_label_window_after_input_window():
    g, ev, pt = table(raw_case())
    for e, c in L.EV.items():
        on = ev[ev.event == e].onset_sec.to_numpy()
        for h in H:
            y = pt[f"y_{e}_{h}"]
            for t, v in zip(pt.t_sec, y):
                hit = ((on > t) & (on <= t + h * 60)).any()
                assert np.isnan(v) or v == hit  # positives only from onsets in (t, t + H]
        assert (pt[f"next_onset_{e}_s"].dropna() > 0).all()
        np.testing.assert_array_equal(pt[f"{c['signal']}_now"], g.set_index("t_sec")[c["signal"]].loc[pt.t_sec])
    assert pt.loc[pt.t_sec == 40 * 60, "y_hypotension_5"].item() == 0  # onset at t itself is in progress, not ahead


def test_censoring_at_end_of_recording():
    g, _, pt = table(raw_case())
    for e, c in L.EV.items():
        s = c["signal"]
        last = g.t_sec[g[f"{s}_raw"].notna() & ~g[f"art_{s}"]].max()
        assert last < g.t_sec.iat[-1] - 20  # each label signal ends before the file
        for h in H:
            # an onset at t + H is only an event once min_duration_s of it is recorded
            y, late = pt[f"y_{e}_{h}"], pt.t_sec + h * 60 > last - c["min_duration_s"] + STEP
            assert y[late].isna().all() and y[~late].notna().all()


def test_saved_splits_disjoint_and_reproducible(monkeypatch, tmp_path):
    if not (PROCESSED / "splits.json").exists():
        pytest.skip("run uv run python -m orew.labels first")
    splits = json.loads((PROCESSED / "splits.json").read_text())
    folds = json.loads((PROCESSED / "folds.json").read_text())
    sets = [set(v) for v in splits.values()]
    assert sum(map(len, sets)) == len(set.union(*sets))  # no subject in two splits
    assert {int(k) for k in folds} == set(splits["train"]) and set(folds.values()) == set(range(CFG["split"]["n_folds"]))
    if (PROCESSED / "pred_table.parquet").exists():
        pt = pd.read_parquet(PROCESSED / "pred_table.parquet", columns=["subjectid", "split"])
        assert pt.groupby("subjectid").split.nunique().max() == 1
        assert all(pt.subjectid[pt.split == k].isin(v).all() for k, v in splits.items())
    monkeypatch.setattr(L, "PROCESSED", tmp_path)
    L.split()
    for f in ("splits.json", "folds.json"):
        assert (tmp_path / f).read_text() == (PROCESSED / f).read_text()
    (tmp_path / "folds.json").write_text("{}")
    with pytest.raises(RuntimeError):
        L.split()
