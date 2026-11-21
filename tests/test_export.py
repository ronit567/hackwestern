import json
import math
import shutil
import subprocess

import pandas as pd
import pytest

from orew import RAW, ROOT
from orew.export import PHASES, SERIES, STEP, export, rftn_ug_h


def load(p):
    def no_nan(c):
        raise ValueError(f"{c} in {p}")

    return json.loads(p.read_text(), parse_constant=no_nan)


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    out = tmp_path_factory.mktemp("web")
    if not export(max_cases=2, out=out):
        pytest.skip("no cached cases in data/raw")
    return out


def test_case_files_match_source(out):
    cases = pd.read_csv(RAW / "cases.csv", encoding="utf-8-sig").set_index("caseid")
    index = load(out / "cases" / "index.json")
    assert len(index) == 2
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
