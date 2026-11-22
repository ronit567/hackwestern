"""The live replay reproduces the locked test predictions, alarms and drivers, and is causal (PLAN.md "Live demo")."""
import json

import numpy as np
import pandas as pd
import pytest

from orew import PROCESSED, REPORTS
from orew.live import E, LiveModel, local_grid

CASES = [287, 1046]  # featured test surgeries: a clean warning and an honest miss


@pytest.fixture(scope="module")
def model():
    if not (PROCESSED / "preds_test.parquet").exists() or not (PROCESSED / "models" / "fusion.pkl").exists():
        pytest.skip("needs the locked test predictions and saved models")
    return LiveModel()


@pytest.fixture(scope="module")
def replays(model):
    return {c: model.replay(c) for c in CASES}


def test_only_test_surgeries(model):
    split = pd.read_parquet(PROCESSED / "pred_table.parquet", columns=["caseid", "split"]).drop_duplicates()
    assert set(model.ids) == set(split.caseid[split.split == "test"])
    s = model.surgeries()
    assert [x["caseid"] for x in s if x["featured"]] == [f["caseid"] for f in json.loads((REPORTS.parent / "web" /
                                                                                           "featured.json").read_text())]
    assert all(x["featured"] for x in s[:4]) and not any(x["featured"] for x in s[4:])


@pytest.mark.parametrize("cid", CASES)
def test_probabilities_match_preds_test(replays, cid):
    p = pd.read_parquet(PROCESSED / "preds_test.parquet", filters=[("caseid", "==", cid)])
    p = p[p[f"eligible_{E}"]].set_index("t_sec")
    live = {s["t_sec"]: s for s in replays[cid]["steps"] if s["scored"]}
    assert set(live) == set(p.index)
    diff = max(abs(live[t][f"p{h}"] - p.at[t, f"p_fusion_{E}_{h}"]) for t in live for h in (5, 10, 15))
    print(f"case {cid}: max |live - preds_test| = {diff:.3g} over {len(live)} times x 3 horizons")
    assert diff < 1e-9
    assert all(s["p5"] is None for s in replays[cid]["steps"] if not s["scored"])
    # the monitor's specialist columns: 10-min snapshot and baseline 1 probabilities and lead_specialist
    for k, col in (("p_snapshot_10", f"p_snapshot_{E}_10"), ("p_baseline1_10", f"p_baseline1_{E}_10")):
        d = max(abs(live[t][k] - p.at[t, col]) for t in live)
        print(f"case {cid}: max |live {k} - preds_test {col}| = {d:.3g}")
        assert d < 1e-9, k
    assert [live[t]["lead"] for t in sorted(live)] == p.loc[sorted(live), f"lead_specialist_{E}"].tolist()


@pytest.mark.parametrize("cid", CASES)
def test_alarms_drivers_and_verdict_match_case_file(replays, cid):
    case = json.loads((REPORTS / "web" / "cases" / f"{cid}.json").read_text())
    want = [a for a in case["alerts"] if a["type"] == E]
    got = [s["alarm"] for s in replays[cid]["steps"] if s["alarm"]]
    assert [round(a["t_sec"] / 60, 2) for a in got] == [a["t_min"] for a in want]
    assert [float(f"{a['p10']:.3g}") for a in got] == [a["p10"] for a in want]
    assert [a["map_now"] for a in got] == [a["now"] for a in want]
    assert [a["drivers"] for a in got] == [a["top_drivers"] for a in want]  # = records_test.jsonl top_drivers_text
    recs = {r["t_sec"]: r[E] for r in map(json.loads, (PROCESSED / "records_test.jsonl").open()) if r["caseid"] == cid}
    assert [a["drivers"] for a in got] == [recs[a["t_sec"]]["top_drivers_text"] for a in got]
    det = {k: v for k, v in case["detection"][E].items() if k not in ("not_warnable", "hours")}
    assert replays[cid]["summary"] == det
    onsets = [o for s in replays[cid]["steps"] for o in s["onsets"]]
    ev = [x for x in case["events"] if x["type"] == E]
    assert [round(o["t_sec"] / 60, 2) for o in onsets] == [x["onset_min"] for x in ev]
    assert [o["lead_min"] for o in onsets] == [x.get("lead_min") for x in ev]


def test_vitals_and_meta(replays):
    r = replays[287]
    assert r["meta"]["first_pred_sec"] == 600 and r["steps"][-1]["t_sec"] == r["meta"]["duration_s"] // 30 * 30
    assert [v["t_sec"] for v in r["vitals"]] == list(range(0, len(r["vitals"]) * 10, 10))
    assert set(r["vitals"][0]) == {"t_sec", "map", "hr", "spo2", "sbp", "dbp", "bt", "rr", "etco2"}
    assert all(any(v[k] is not None for v in r["vitals"]) for k in ("sbp", "dbp", "bt", "rr", "etco2"))
    # the MAP tile reads the cleaned MAP the table and banner show (map_now), not the raw zero line before the catheter
    assert all(r["vitals"][s["t_sec"] // 10]["map"] == s["map_now"] for s in r["steps"])
    assert min(v["map"] for v in r["vitals"] if v["map"] is not None) >= 20
    json.dumps(r, allow_nan=False)


def test_causal_when_future_rows_are_deleted(model, replays, monkeypatch):
    """Probabilities and alarms at t are unchanged when every grid row after t is deleted (alarm at 31 min, the last
    ticks before both onsets at 2128 s and 2346 s, and mid-surgery)."""
    g = local_grid(287)
    full = {s["t_sec"]: s for s in replays[287]["steps"]}
    for t in [1860, 2100, 2340, 4500]:
        monkeypatch.setattr(model, "grid", lambda c, t=t: g[g.t_sec <= t])
        cut = model.replay(287)
        last = cut["steps"][-1]
        assert last["t_sec"] == t and last["scored"]
        keys = ("p5", "p10", "p15", "map_now", "alarm", "p_snapshot_10", "p_baseline1_10", "lead")
        assert {k: last[k] for k in keys} == {k: full[t][k] for k in keys}
        assert cut["vitals"] == replays[287]["vitals"][: t // 10 + 1]  # every vital up to t, sbp..etco2 included
    assert full[1860]["alarm"] is not None
