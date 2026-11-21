import json
import math
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from orew import PROCESSED, RAW, ROOT
from orew.evaluate import EVENTS, HH, HS, alarms
from orew.export import PHASES, SERIES, STEP, export, labels, load_risk, rftn_ug_h, risk_sources


def load(p):
    def no_nan(c):
        raise ValueError(f"{c} in {p}")

    return json.loads(p.read_text(), parse_constant=no_nan)


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    out = tmp_path_factory.mktemp("web")
    if not export(max_cases=4, out=out):
        pytest.skip("no cached cases in data/raw")
    return out


def test_case_files_match_source(out):
    cases = pd.read_csv(RAW / "cases.csv", encoding="utf-8-sig").set_index("caseid")
    index = load(out / "cases" / "index.json")
    assert len(index) == 4
    for entry in index:
        c = entry["caseid"]
        case, row = load(out / "cases" / f"{c}.json"), cases.loc[c]
        src = pd.read_parquet(RAW / f"{c:04d}.parquet")
        src["rftn_ug_h"] = rftn_ug_h(src)
        # expected: last finite 2 s value in each (t - 10, t] bin, None if the bin is all NaN
        exp = {s: {} for s in SERIES}
        for r in src.itertuples():
            t = math.ceil(r.t_sec / STEP) * STEP
            for s in SERIES:
                exp[s].setdefault(t, None)
                if not math.isnan(getattr(r, s)):
                    exp[s][t] = round(getattr(r, s), 1)
        bins = sorted(exp["map"])
        assert case["t_min"] == [round((t - row.casestart) / 60, 2) for t in bins]
        for s in SERIES:
            assert case["series"][s] == [exp[s][t] for t in bins], s
        assert any(v is None for v in case["series"]["nibp_map"])  # NIBP is sparse, so NaN -> null is exercised
        for k in PHASES:
            assert case["phases_min"][k] == pytest.approx((row[k] - row.casestart) / 60, abs=0.01)


def test_cohort_json(out):
    coh = load(out / "cohort.json")
    n = coh["n_cached"]
    # <= not ==: download.py may add cases while this test runs
    assert 0 < n <= len(list(RAW.glob("[0-9][0-9][0-9][0-9].parquet"))) and n <= coh["n_cohort"]
    assert sum(coh["sex"].values()) == sum(coh["asa"].values()) == coh["emergency"]["of"] == n
    assert {s["name"] for s in coh["signals"]} >= set(SERIES) - {"rftn_ug_h"}  # derived, not a raw column
    assert all(0 <= s["pct_time"] <= s["pct_cases"] <= 100 for s in coh["signals"])


def test_label_fields(out):
    ev, split_of = labels()
    if ev is None or not split_of:
        pytest.skip("labels.py has not run")
    for entry in load(out / "cases" / "index.json"):
        case = load(out / "cases" / f"{entry['caseid']}.json")
        assert case["split"] == entry["split"] in {"train", "val", "test", None}
        for e in case.get("events", []):
            assert e["type"] in {"hypotension", "hypoxemia"} and e["onset_min"] <= e["end_min"]
    coh = load(out / "cohort.json")
    assert sum(s["n_cases"] for s in coh["splits"].values()) == coh["n_cohort"]
    assert all(e["n_cases"] <= coh["n_labelled"] <= coh["n_cached"] for e in coh["events"].values())


def risk_cases(out):
    cases = [load(out / "cases" / f"{e['caseid']}.json") for e in load(out / "cases" / "index.json")]
    if not any("risk" in c for c in cases):
        pytest.skip("models.py has not run")
    return cases


def test_risk_and_alerts_match_preds_and_evaluate(out):
    """Risk = the case's own-split fusion preds (null where ineligible); alerts = evaluate's simulator over the whole
    split (rows sorted by caseid, t_sec, eligible and finite, threshold.json), restricted to the case."""
    cases, info = risk_cases(out), pd.read_csv(RAW / "cases.csv", encoding="utf-8-sig").set_index("caseid")
    thr = json.loads((PROCESSED / "models" / "threshold.json").read_text())
    src, sims = risk_sources(), {}
    assert cases[0]["split"] == "val" and cases[0]["events"]  # the explorer opens on a val surgery with events
    for c in (c for c in cases if "risk" in c):
        if c["split"] not in sims:
            df = pd.read_parquet(src[c["split"]]).sort_values(["caseid", "t_sec"], ignore_index=True)
            for e in thr:
                mon = df[df[f"eligible_{e}"] & df[f"p_fusion_{e}_{HH}"].notna()]
                a = alarms(mon[f"p_fusion_{e}_{HH}"].to_numpy(), mon.caseid.to_numpy(), mon.t_sec.to_numpy(), thr[e]["threshold"])
                df[f"alarm_{e}"] = pd.Series(a, mon.index).reindex(df.index, fill_value=False)
            sims[c["split"]] = df
        d, t0 = sims[c["split"]].query(f"caseid == {c['caseid']}"), info.casestart[c["caseid"]]
        for e, r in c["risk"].items():
            assert r["threshold"] == thr[e]["threshold"]
            assert r["t_min"] == ((d.t_sec - t0) / 60).round(2).tolist()
            for h in HS:
                p = d[f"p_fusion_{e}_{h}"].where(d[f"eligible_{e}"]).to_numpy()
                got = np.array(r[f"p{h}"], dtype=float)  # None -> NaN
                assert len(got) == len(r["t_min"]) and np.array_equal(np.isnan(got), np.isnan(p))
                np.testing.assert_allclose(got, p, rtol=5e-3)
            ok = [i for i, v in enumerate(r["p15"]) if v is not None]
            assert all(r["p5"][i] <= r["p10"][i] <= r["p15"][i] for i in ok)
            assert [a["t_min"] for a in c["alerts"] if a["type"] == e] == ((d.t_sec[d[f"alarm_{e}"]] - t0) / 60).round(2).tolist()
        for a in c["alerts"]:
            assert a["p10"] >= float(f"{thr[a['type']]['threshold']:.3g}")  # both rounded to 3 significant digits
            assert a["top_drivers"] is None or all(isinstance(x, str) for x in a["top_drivers"])


def test_load_risk_gates_test_split_and_stale_records(tmp_path):
    """Synthetic processed dir: test-split preds are used only with the lock; records drive alerts only while their
    p10 matches the preds."""
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "threshold.json").write_text(json.dumps({e: {"threshold": 0.5, "chosen_on": "val"} for e in EVENTS}))
    for f, split, c in [("preds_oof", "train", 1), ("preds_val", "val", 2), ("preds_test", "test", 3)]:
        pd.DataFrame({"caseid": [c], "t_sec": [600], "split": [split], **{f"eligible_{e}": [True] for e in EVENTS},
                      **{f"p_fusion_{e}_{h}": [0.25] for e in EVENTS for h in HS}}).to_parquet(tmp_path / f"{f}.parquet")
    rec = {"caseid": 2, "t_sec": 600, "hypotension": {"p10": 0.25, "top_drivers_text": ["a", "b", "c"]}, "hypoxemia": None}
    (tmp_path / "records_val.jsonl").write_text(json.dumps(rec) + "\n")
    lock = tmp_path / "test.lock"
    by_case, drivers, _ = load_risk({1, 2, 3}, tmp_path, lock)
    assert set(by_case) == {1, 2} and drivers[2, 600]["hypotension"] == ["a", "b", "c"]
    lock.touch()
    assert set(load_risk({1, 2, 3}, tmp_path, lock)[0]) == {1, 2, 3}
    rec["hypotension"]["p10"] = 0.3  # records from an older model
    (tmp_path / "records_val.jsonl").write_text(json.dumps(rec) + "\n")
    assert load_risk({2}, tmp_path, lock)[1] == {}


def test_page_script_parses():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    subprocess.run([node, "--check", str(ROOT / "web" / "app.js")], check=True)


def test_rftn_ug_h_sums_both_pumps():
    nan = float("nan")
    df = pd.DataFrame({"rftn_rate": [1.0, nan, 2.0, nan], "rftn50_rate": [nan, 1.0, 1.0, nan]})
    assert rftn_ug_h(df).tolist()[:3] == [20.0, 50.0, 90.0] and math.isnan(rftn_ug_h(df).iat[3])
    assert rftn_ug_h(df.drop(columns="rftn50_rate")).iat[0] == 20.0
