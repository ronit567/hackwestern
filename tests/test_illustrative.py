import json

from orew import ROOT
from orew.illustrative import OUT, build

D = build()
TRIPLES = []  # (where, [e, lo, hi]) for every metric in the file


def walk(o, where=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{where}/{k}")
    elif isinstance(o, list) and len(o) == 3 and all(isinstance(v, float | int) for v in o) and "horizons" not in where:
        TRIPLES.append((where, o))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{where}/{i}")


walk(D)


def test_committed_file_is_labelled_and_fresh():
    assert json.loads(OUT.read_text()) == D  # rerun `uv run python -m orew.illustrative` after editing the generator
    assert D["illustrative"] is True and D["event"] == "hypoxemia"
    assert "made up" in D["note"] and "not model results" in D["note"]


def test_intervals_contain_estimates():
    assert len(TRIPLES) > 500
    for where, (e, lo, hi) in TRIPLES:
        assert lo <= e <= hi, where
        if not any(k in where for k in ("false_per_true", "median_lead", "alarms_per_hour")):
            assert 0 <= lo and hi <= 1, where
    for H, byM in D["overall"].items():
        for m, d in byM.items():
            if "event_recall" in d:
                assert d["pct_warned_5min"][0] <= d["event_recall"][0], (H, m)


def test_counts_add_up():
    n = D["n"]
    for H in D["bins"]:
        rows = D["bins"][H]
        assert all(0 < b["n_pos"] <= b["n"] for b in rows + D["bins_excl_flagged"][H])
        assert sum(b["n"] for b in rows) <= n["times"]
        for groups in D["subgroups"][H].values():
            assert sum(g["n"] for g in groups) == n["times"]
            assert sum(g["n_pos"] for g in groups) >= sum(b["n_pos"] for b in rows)
            assert all(0 < g["n_pos"] <= g["n"] for g in groups)
        for pts in D["reliability"][H].values():
            assert sum(p["n"] for p in pts) <= n["times"] and all(0 <= p["obs"] <= 1 for p in pts)


def test_modest_not_spectacular():
    head = D["overall"][str(D["headline_horizon_min"])]
    assert 0.80 <= head["fusion"]["auroc"][0] <= 0.85 and 0.65 <= head["baseline1"]["auroc"][0] <= 0.75
    for H, byM in D["overall"].items():
        assert byM["fusion"]["auroc"][0] > byM["baseline1"]["auroc"][0], H
        assert byM["fusion"]["auroc"][2] - byM["fusion"]["auroc"][1] >= 0.1, H  # wide-ish
        assert D["ablation"][H]["with_drugs"]["auroc"] == byM["snapshot"]["auroc"]
    assert set(D["alert_threshold"]) == set(D["models"])


def test_pipeline_never_reads_it():
    for p in (ROOT / "src" / "orew").glob("*.py"):
        if p.name != "illustrative.py":
            assert "illustrative" not in p.read_text(), p
