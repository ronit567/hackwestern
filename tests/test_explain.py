import json
import math

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from orew import explain
from orew.explain import EVENTS, HEAD, HORIZONS, describe

try:
    from orew.features import FEATURES
except ImportError:  # features.py not built yet
    FEATURES = ["map", "map_slope_5m", "ppf_rate_delta_5m", "beat_pp", "spo2_slope_3m", "phase", "phase_emergence"]


@pytest.mark.parametrize("v", [math.nan, 0.0, 1.0, -1.5, 3.0])
def test_every_feature_has_a_template(v):
    for f in FEATURES:
        assert explain._template(f)[1] is not None, f"{f} falls back to the generic text"
        d = describe(f, v)
        assert d is None or isinstance(d, str) and d  # None: an implausible level, skipped as a driver


def test_template_text():
    assert describe("map_slope_5m", -1.5).startswith("MAP falling 1.5 mmHg/min")
    assert describe("ppf_rate_delta_5m", 40) == "propofol rate just increased (+40 mL/h in 5 min)"
    assert describe("beat_pp", 28) == "pulse pressure 28 mmHg"
    assert describe("spo2_slope_3m", -0.8).startswith("SpO2 falling 0.8%/min")
    assert describe("phase", 3) == "emergence phase"
    assert describe("not_a_feature_x", 2) == "not a feature x 2"
    assert describe("bt", np.float32("nan")) == "temperature: no recent data"


def test_implausible_levels_are_not_quoted():
    assert describe("dbp_min_10m", -30) is None and describe("hr", 0) is None and describe("bt", 5.6) is None
    assert describe("map_mean_5m", 250) is None and describe("sbp", -81) is None
    assert describe("dbp_min_10m", 40) == "diastolic BP dipped to 40 mmHg over last 10 min"
    assert describe("hr", 80) == "heart rate 80 bpm"
    assert describe("map_slope_5m", -30).startswith("MAP falling 30")  # slopes are not range-checked


def _fake(tmp_path, n=200):
    rng = np.random.default_rng(0)
    keys = pd.DataFrame({"caseid": np.repeat([1, 2], n // 2), "t_sec": np.tile(np.arange(n // 2) * 30 + 600, 2)})
    cols = ["map", "map_slope_5m", "beat_pp", "spo2", "spo2_slope_3m", "phase"]
    feat = keys.assign(**{c: rng.normal(size=n).astype("float32") for c in cols})
    feat.loc[::7, "map"] = np.nan
    feat.to_parquet(tmp_path / "features.parquet")
    preds = keys.copy()
    (tmp_path / "models").mkdir()
    for k, e in enumerate(EVENTS):
        preds[f"eligible_{e}"] = rng.random(n) > 0.3
        for h in HORIZONS:
            preds[f"p_fusion_{e}_{h}"] = rng.random(n)
        preds[f"lead_specialist_{e}"] = "snapshot"
        use = cols[: 4 + k]  # different feature subsets per event
        y = (feat[use[0]].fillna(0) + feat[use[1]] > 0).astype(int)
        xgb.XGBClassifier(n_estimators=10, max_depth=3).fit(feat[use], y).save_model(
            tmp_path / "models" / f"snapshot_{e}_{HEAD}.json")
    preds.to_parquet(tmp_path / "preds_val.parquet")
    return feat, preds


def test_records(tmp_path):
    feat, preds = _fake(tmp_path)
    recs, _ = explain.records("val", tmp_path)
    json.dumps(recs)  # serialisable
    assert len(recs) == len(preds)
    skipped = 0
    for e in EVENTS:
        b = xgb.Booster()
        b.load_model(tmp_path / "models" / f"snapshot_{e}_{HEAD}.json")
        phi = pd.DataFrame(b.predict(xgb.DMatrix(feat[b.feature_names]), pred_contribs=True)[:, :-1],
                           columns=b.feature_names)
        for i, r in enumerate(recs):
            assert set(r) == {"caseid", "t_sec", *EVENTS}
            assert (r["caseid"], r["t_sec"]) == tuple(preds.loc[i, ["caseid", "t_sec"]])
            blk = r[e]
            if not preds.at[i, f"eligible_{e}"]:
                assert blk is None
                continue
            assert set(blk) == {"p5", "p10", "p15", "lead_specialist", "top_drivers", "top_drivers_text"}
            assert all(isinstance(blk[f"p{h}"], float) for h in HORIZONS)
            assert blk["lead_specialist"] == "snapshot"
            d = blk["top_drivers"]
            assert len(d) <= 3 and len(blk["top_drivers_text"]) == len(d)
            assert all(isinstance(s, str) for s in d + blk["top_drivers_text"])
            # the first 3 risk-raising features by SHAP whose text quotes a plausible value
            row = phi.loc[i]
            want = [f for f in row.index[np.argsort(-row.to_numpy(), kind="stable")]
                    if row[f] > 0 and describe(f, feat.at[i, f]) is not None][:3]
            assert d == want
            assert blk["top_drivers_text"] == [describe(f, feat.at[i, f]) for f in d]
            skipped += sum(row[f] > 0 and describe(f, feat.at[i, f]) is None for f in row.index)
    assert skipped  # N(0, 1) features put MAP below 30 mmHg, so implausible drivers were met and skipped
