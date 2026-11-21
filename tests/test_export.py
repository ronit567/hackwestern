import json
import math

import pandas as pd
import pytest

from orew import RAW
from orew.export import PHASES, SERIES, STEP, export


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
    n = len([p for p in RAW.glob("[0-9][0-9][0-9][0-9].parquet")])
    assert coh["n_cached"] == n <= coh["n_cohort"]
    assert sum(coh["sex"].values()) == sum(coh["asa"].values()) == coh["emergency"]["of"] == n
    assert {s["name"] for s in coh["signals"]} >= set(SERIES)
    assert all(0 <= s["pct_time"] <= s["pct_cases"] <= 100 for s in coh["signals"])
