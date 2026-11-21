"""orew.models on synthetic data where y depends on one feature: leakage #8, #9, #10."""
import numpy as np
import pandas as pd
import pytest

from orew import models


def synth(seed=0, n_subj=40, n_t=60):
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(n_subj), n_t)
    df = pd.DataFrame({"caseid": s + 1000, "subjectid": s, "t_sec": np.tile(600 + 30 * np.arange(n_t), n_subj)})
    df["split"] = np.where(s < 28, "train", np.where(s < 34, "val", "test"))
    for e, base in (("hypotension", 80), ("hypoxemia", 97)):
        slope, cur = models.SLOPE[e], models.CUR[e]
        df[slope] = rng.normal(size=len(df)).astype("float32")
        df[cur] = base + rng.normal(size=len(df)) + 50 * (df.split != "train")  # val/test shifted: train stats must differ
        df.loc[rng.random(len(df)) < 0.05, cur] = np.nan
        df[f"eligible_{e}"] = rng.random(len(df)) > 0.1
        df[f"flag_{e}"] = False
        for h in models.HORIZONS:
            df[f"y_{e}_{h}"] = (df[slope] + 0.3 * rng.normal(size=len(df)) < -1 + h / 30).astype(float)
            df.loc[df.t_sec > df.t_sec.max() - h * 60, f"y_{e}_{h}"] = np.nan  # censored tail
        df[f"next_onset_{e}_s"] = np.nan
    df["ppf_rate"] = rng.random(len(df)).astype("float32")
    df["phase_maintenance"] = (df.t_sec > 1200).astype("float32")
    folds = {i: i % 5 for i in range(28)}
    return df, folds, ["map_slope_5m", "spo2_slope_5m", "ppf_rate", "phase_maintenance"], ["ppf_rate"]


@pytest.fixture(scope="module")
def fitted():
    df, folds, feats, drugs = synth()
    calls = []
    real = models.crossfit
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(models.XGB, "n_estimators", 5)
        mp.setattr(models, "crossfit", lambda fit, pred, d, f: calls.append(d) or real(fit, pred, d, f))
        res = models.run(df, folds, feats, drugs, ("val", "test"))
    return df, folds, feats, calls, res


def test_crossfit_never_predicts_with_a_model_that_saw_the_subject():
    df, folds, _, _ = synth()
    tr = df[df.split == "train"]
    seen = []
    fit = lambda d: seen.append(set(d.subjectid)) or len(seen) - 1  # "model" = index of the subjects it saw

    def pred(m, d):
        assert not seen[m] & set(d.subjectid)
        return np.full(len(d), m)

    oof, fold, full = models.crossfit(fit, pred, tr, folds)
    assert np.isfinite(oof).all() and (fold == tr.subjectid.map(folds)).all()
    for s, m in zip(tr.subjectid, oof.astype(int)):
        assert s not in seen[int(m)]
    assert seen[full] == set(tr.subjectid)


def test_outputs_shape_and_oof_folds(fitted):
    df, folds, _, _, (out, saved, thr) = fitted
    assert set(out) == {"train", "val", "test"}
    assert (out["train"].fold == out["train"].subjectid.map(folds)).all()
    for s, d in out.items():
        assert len(d) == (df.split == s).sum()
        for e in models.EVENTS:
            el = d[f"eligible_{e}"]
            for m in models.MODELS:
                for h in models.HORIZONS:
                    p = d[f"p_{m}_{e}_{h}"]
                    assert p[el].between(0, 1).all() and p[~el].isna().all()
            if s != "train":
                assert d.loc[el, f"lead_specialist_{e}"].isin(models.GROUPS).all()
    # the specialist learnt the planted signal
    v = out["val"][out["val"].eligible_hypotension & out["val"].y_hypotension_10.notna()]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(v.y_hypotension_10, v.p_snapshot_hypotension_10) > 0.8


def test_stacker_trains_on_oof_predictions_only(fitted):
    df, _, feats, calls, (out, saved, _) = fitted
    stack = [d for d in calls if "trend" in d.columns]
    assert len(stack) == 2 * len(models.EVENTS) * len(models.HORIZONS)
    e, h = models.EVENTS[0], models.HORIZONS[0]
    tr = out["train"][out["train"][f"eligible_{e}"]]
    x = stack[0]  # first stacker: (first event, first horizon, fusion)
    assert np.allclose(x.snapshot, models.logit(tr[f"p_snapshot_{e}_{h}"].to_numpy()))
    assert np.allclose(x.trend, models.logit(tr[f"p_baseline2_{e}_{h}"].to_numpy()))
    rows = df[(df.split == "train") & df[f"eligible_{e}"]]
    insample = saved["snapshot"][f"snapshot_{e}_{h}"].predict_proba(rows[feats])[:, 1]
    assert not np.allclose(x.snapshot, models.logit(insample))


def test_imputer_and_scaler_fit_on_train_only(fitted):
    df, *_, (out, saved, _) = fitted
    for e in models.EVENTS:
        h = models.HEAD
        cols = [models.CUR[e], models.SLOPE[e]]
        tr = df[(df.split == "train") & df[f"eligible_{e}"] & df[f"y_{e}_{h}"].notna()][cols]
        imp, sc, _ = saved["fusion"]["models"][f"{e}_{h}"]["baseline2"]
        assert np.allclose(imp.statistics_, tr.median())
        assert np.allclose(sc.mean_, tr.fillna(tr.median()).mean())
        assert np.allclose(sc.scale_, tr.fillna(tr.median()).std(ddof=0))
        assert not np.isclose(sc.mean_[0], df[cols[0]].mean())


def test_threshold_chosen_on_val_only(fitted):
    df, folds, feats, _, (out, _, thr) = fitted
    for e in models.EVENTS:
        assert thr[e]["chosen_on"] == "val"
        assert thr[e]["threshold"] == models.choose_threshold(out["val"], f"p_fusion_{e}_{models.HEAD}")
    # scrambling test labels and features changes nothing that was fit or chosen
    d2 = df.copy()
    t = d2.split == "test"
    rng = np.random.default_rng(1)
    for c in feats + [c for c in d2 if c.startswith("y_")]:
        d2.loc[t, c] = rng.permutation(d2.loc[t, c].to_numpy())
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(models.XGB, "n_estimators", 5)
        out2, _, thr2 = models.run(d2, folds, feats, ["ppf_rate"], ("val", "test"))
    assert thr2 == thr
    pd.testing.assert_frame_equal(out2["val"], out["val"])

