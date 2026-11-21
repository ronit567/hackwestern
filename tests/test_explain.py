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
        assert isinstance(describe(f, v), str) and describe(f, v)


def test_template_text():
    assert describe("map_slope_5m", -1.5).startswith("MAP falling 1.5 mmHg/min")
    assert describe("ppf_rate_delta_5m", 40) == "propofol rate just increased (+40 mL/h in 5 min)"
    assert describe("beat_pp", 28) == "pulse pressure 28 mmHg"
    assert describe("spo2_slope_3m", -0.8).startswith("SpO2 falling 0.8%/min")
    assert describe("phase", 3) == "emergence phase"
    assert describe("not_a_feature_x", 2) == "not a feature x 2"


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
            s = phi.loc[i, d].to_numpy()
            assert (s > 0).all() and (np.diff(s) <= 0).all()  # positive, ordered by SHAP
            rest = phi.loc[i].drop(d)
            assert len(d) == 3 or (rest <= 0).all()  # fewer than 3 only when no other feature pushes risk up
            if d:
                assert s[0] == phi.loc[i].max()
                assert blk["top_drivers_text"][0] == describe(d[0], feat.at[i, d[0]])
