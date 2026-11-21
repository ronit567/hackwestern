import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from orew import REPORTS
from orew import evaluate as E


def _ct(S, B=20):
    return np.ascontiguousarray(E.boot_counts(S, B).T)


def test_perfect_random_and_sklearn_match():
    rng = np.random.default_rng(0)
    n, S = 5000, 40
    y, s = rng.random(n) < 0.2, rng.integers(0, S, n)
    Ct = _ct(S)
    perfect = E.disc(y + 0.5 * rng.random(n), y, s, Ct)
    assert np.allclose(perfect[[0, 1, 3]], 1)  # auroc, auprc, sens@90spec = 1 in every resample
    assert abs(E.disc(rng.random(n), y, s, Ct)[0, 0] - 0.5) < 0.05
    p = np.round(rng.random(n) * 0.5 + 0.3 * y, 2)  # ties on purpose
    v = E.disc(p, y, s, Ct)[:, 0]
    fpr, tpr, _ = roc_curve(y, p)
    assert v[0] == pytest.approx(roc_auc_score(y, p))
    assert v[1] == pytest.approx(average_precision_score(y, p))
    assert v[2] == pytest.approx(np.mean((p - y) ** 2), rel=1e-5)
    assert v[3] == pytest.approx(tpr[fpr <= 0.1].max())


def test_bootstrap_resamples_by_subject():
    rng = np.random.default_rng(1)
    S, n = 15, 600
    C = E.boot_counts(S, 30)
    assert (C[0] == 1).all() and (C.sum(1) == S).all()
    s = rng.integers(0, S, n)
    y = rng.random(n) < 0.3
    p = rng.random(n) + 0.3 * y
    v = E.disc(p, y, s, np.ascontiguousarray(C.T))
    for b in [1, 7]:  # resample b == every row of each drawn subject, repeated as often as the subject was drawn
        idx = np.concatenate([np.flatnonzero(s == j) for j in range(S) for _ in range(int(C[b, j]))])
        assert v[0, b] == pytest.approx(roc_auc_score(y[idx], p[idx]))
        assert v[1, b] == pytest.approx(average_precision_score(y[idx], p[idx]))


def test_alarm_refractory_and_rearm():
    t = np.r_[0, 30, 60, 90, 330, 360, 0]
    p = np.r_[.6, .65, .72, .75, .5, .5, .6]
    case = np.r_[0, 0, 0, 0, 0, 0, 1]
    # 0: fires; 30: refractory; 60: rearm (>= .6 + .1); 90: refractory (< .82); 330: 270 s after 60 -> quiet;
    # 360: 300 s after 60 -> fires; new case resets the clock
    assert E.alarms(p, case, t, 0.5).tolist() == [1, 0, 1, 0, 0, 1, 1]
    assert not E.alarms(np.r_[np.nan, .4], np.r_[0, 0], np.r_[0, 30], 0.5).any()


def test_lead_time_recall_and_false_alarms():
    t = np.arange(600, 3001, 30)
    d = pd.DataFrame({"caseid": 0, "t_sec": t, "s": 0})
    on = np.r_[1980, 2700]
    d["nxt"] = np.r_[on, np.nan][np.searchsorted(on, t, "right")] - t
    p = np.where(np.isin(t, [1500, 900]), 0.9, 0.1)  # 900: next onset 1080 s away -> false; 1500: 480 s before 1980
    ev = pd.DataFrame({"caseid": 0, "onset_sec": on, "s": 0})
    m, log = E.alert_metrics(d, p, 0.5, ev, E.boot_counts(1, 5))
    assert log == {"alarms": 2, "true_alarms": 1, "events": 2, "warned": 1, "hours": pytest.approx(len(t) * 30 / 3600)}
    assert m["median_lead_min"][0] == pytest.approx(8.0)
    assert m["event_recall"][0] == pytest.approx(0.5) and m["pct_warned_5min"][0] == pytest.approx(0.5)
    assert m["false_per_true_alarm"][0] == pytest.approx(1.0)
    assert m["alarms_per_hour"][0] == pytest.approx(2 / log["hours"])


def test_bins_assign_and_labels():
    x = np.r_[64.9, 65, 69.99, 70, 80, 999, np.nan]
    assert E.bin_index(x, [[65, 70], [70, 75], [75, 80], [80, 1000]]).tolist() == [-1, 0, 0, 1, 3, 3, -1]
    assert [E.bin_label("hypotension", 65, 70, False), E.bin_label("hypotension", 80, 1000, True)] == ["65-70", ">80"]
    assert [E.bin_label("hypoxemia", 90, 95, False), E.bin_label("hypoxemia", 95, 101, True)] == ["90-94", ">=95"]


def synth(n=24, seed=0):
    """Synthetic preds / events / cases in the preds_{split}.parquet schema."""
    rng = np.random.default_rng(seed)
    rows, evs = [], []
    for c in range(n):
        t = np.arange(600, 4801, 30)
        d = {"caseid": c, "subjectid": 1000 + c // 2 * 2, "t_sec": t, "split": "val",  # 2 cases per subject
             "map_now": rng.uniform(60, 95, len(t)), "spo2_now": rng.integers(89, 101, len(t)).astype(float)}
        for e in E.EVENTS:
            on = np.sort(rng.choice(np.arange(700, 4800, 30), 3, replace=False))
            evs += [(c, e, o) for o in on]
            nxt = np.r_[on, np.nan][np.searchsorted(on, t, "right")] - t
            el = rng.random(len(t)) < 0.95
            d |= {f"eligible_{e}": el, f"flag_{e}": rng.random(len(t)) < 0.1, f"next_onset_{e}_s": nxt}
            for h in E.HS:
                y = np.where(t + h * 60 > 4800, np.nan, (nxt <= h * 60).astype(float))
                d[f"y_{e}_{h}"] = y
                for k, m in enumerate(["baseline1", "snapshot", "snapshot_nodrug", "fusion"]):
                    z = rng.normal(size=len(t)) + k * 0.5 * (nxt <= h * 60) - 2
                    d[f"p_{m}_{e}_{h}"] = np.where(el, 1 / (1 + np.exp(-z)), np.nan)
        rows.append(pd.DataFrame(d))
    cases = pd.DataFrame({"caseid": range(n), "age": rng.uniform(18, 90, n), "sex": rng.choice(["M", "F"], n),
                          "asa": rng.choice([1.0, 2.0, np.nan], n), "department": "General surgery",
                          "emop": rng.integers(0, 2, n)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(evs, columns=["caseid", "event", "onset_sec"]), cases


def test_evaluate_end_to_end_contract_and_json():
    df, ev, cases = synth()
    thr = {e: {"threshold": 0.4, "alarms_per_hour_target": 2.0, "chosen_on": "val"} for e in E.EVENTS}
    M, md = E.evaluate(df, df, ev, cases, thr, 20, "val")
    json.dumps(M, allow_nan=False)  # raises on NaN / inf
    assert M["split"] == "val" and M["final_model"] == "fusion"
    assert M["models"] == ["baseline1", "snapshot", "fusion"]
    assert M["alerts"]["threshold"]["hypotension"]["fusion"] == 0.4
    o = M["overall"]["hypotension"]["10"]["fusion"]
    assert set(E.DISC + E.ALERT) <= set(o) and len(o["auroc"]) == 3
    assert set(E.DISC) <= set(M["overall_excl_flagged"]["hypoxemia"]["5"]["baseline1"])
    b = M["bins"]["hypotension"]["10"]
    assert [x["label"] for x in b] == ["65-70", "70-75", "75-80", ">80"]
    el = df.eligible_hypotension & df.y_hypotension_10.notna()
    assert b[0]["n"] == int((el & (df.map_now >= 65) & (df.map_now < 70)).sum())
    assert "beats_baseline1" in b[0]["models"]["fusion"]
    assert [x["label"] for x in M["bins_excl_flagged"]["hypoxemia"]["15"]] == ["90-94", ">=95"]
    assert set(M["subgroups"]["hypoxemia"]["10"]) == set(E.ATTRS)
    assert len(M["reliability"]["hypotension"]["10"]["fusion"]) == 10
    assert M["ablation"]["hypotension"]["10"]["model"] == "snapshot"
    assert M["n"]["hypotension"]["subjects"] == 12
    assert "## Value bins" in md and "| baseline1 |" in md  # leakage #13


def test_saved_report_has_bin_tables_with_baseline1():
    """Leakage #13 on the real outputs; skips until evaluate has run."""
    f = REPORTS / "val_report.md"
    if not f.exists():
        pytest.skip("reports/val_report.md not built yet")
    txt = f.read_text()
    assert "## Value bins" in txt and "| baseline1 |" in txt
