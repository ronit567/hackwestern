"""Phase 3: baselines, XGBoost snapshot specialist, fusion stacker + isotonic, val alert threshold.

uv run python -m orew.models            # train on train; write preds_oof, preds_val, models/, threshold.json
uv run python -m orew.models --test     # also preds_test, once: needs `pytest -q` green and no reports/TEST_LOCK
"""
import argparse
import json
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from orew import CFG, PROCESSED, REPORTS, ROOT
from orew.evaluate import choose_threshold  # the same p10 alert simulator evaluate reports with

EVENTS = ["hypotension", "hypoxemia"]
HORIZONS = CFG["windows"]["horizons_min"]
HEAD = CFG["windows"]["headline_horizon_min"]
CUR = {"hypotension": "map_now", "hypoxemia": "spo2_now"}
SLOPE = {"hypotension": "map_slope_5m", "hypoxemia": "spo2_slope_5m"}
MODELS = ["baseline1", "baseline2", "snapshot", "snapshot_nodrug", "fusion", "fusion_nodrug"]
XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, tree_method="hist", n_jobs=-1, random_state=CFG["seed"])
GROUPS = ["snapshot", "trend", "current_value", "phase"]  # lead_specialist values; stacker inputs in this order
LOCK = REPORTS / "TEST_LOCK"
OUT = PROCESSED / "models"


def lr(scale=True):
    # imputer + scaler sit inside the pipeline, so each fit learns them from its own training rows only (leakage #8)
    return make_pipeline(SimpleImputer(strategy="median"), *[StandardScaler()] * scale, LogisticRegression(max_iter=1000))


def crossfit(fit, predict, df, folds):
    """fit(rows) -> model, predict(model, rows) -> p. Folds by subjectid -> (OOF p, fold per row, model fit on all df)."""
    fold = df.subjectid.map(folds).to_numpy(float)
    assert not np.isnan(fold).any(), "train subject without a fold"
    oof = np.full(len(df), np.nan)
    for k in np.unique(fold):
        held = fold == k
        tr = df[~held]
        assert not set(tr.subjectid) & set(df.subjectid[held]), "leakage #9: fold model saw a held-out subject"
        oof[held] = predict(fit(tr), df[held])
    return oof, fold, fit(df)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run(df, folds, feats, drugs, splits=("val",)):
    """df = pred_table joined with features. Fits on split == train only. Returns ({train|val|test: preds frame}, saved
    models, thresholds). The train frame holds OOF predictions (fold models never saw the row's subject)."""
    phase = [c for c in feats if c.startswith("phase_")]
    nodrug = [c for c in feats if c not in set(drugs)]
    base = [c for c in df if c not in set(feats) or c in CUR.values()]
    out = {s: df.loc[df.split == s, base].copy() for s in ("train", *splits)}
    src = {s: df[df.split == s] for s in out}
    out["train"]["fold"] = out["train"].subjectid.map(folds)
    saved = {"snapshot": {}, "fusion": {"inputs": GROUPS[:3] + phase, "phase": phase, "models": {}}}
    for e in EVENTS:
        el = {s: d[d[f"eligible_{e}"].astype(bool)] for s, d in src.items()}
        for s in out:
            if s != "train":
                out[s][f"lead_specialist_{e}"] = None
        for h in HORIZONS:
            y = f"y_{e}_{h}"
            P, keep = {s: {} for s in out}, {}

            def fit_on(make, cols):
                return lambda d: make().fit(d.loc[d[y].notna(), cols], d.loc[d[y].notna(), y].astype(int))

            spec = {"baseline1": (lambda: lr(False), [CUR[e]]), "baseline2": (lr, [CUR[e], SLOPE[e]]),
                    "snapshot": (lambda: XGBClassifier(**XGB), feats), "snapshot_nodrug": (lambda: XGBClassifier(**XGB), nodrug)}
            for m, (make, cols) in spec.items():
                pred = lambda mod, d, cols=cols: mod.predict_proba(d[cols])[:, 1]
                P["train"][m], _, keep[m] = crossfit(fit_on(make, cols), pred, el["train"], folds)
                for s in splits:
                    P[s][m] = pred(keep[m], el[s])
            for m in ("snapshot", "snapshot_nodrug"):
                saved["snapshot"][f"{m}_{e}_{h}"] = keep[m]

            # stacker inputs: OOF base predictions on train (leakage #9), full-train base models on val/test
            for m, snap in (("fusion", "snapshot"), ("fusion_nodrug", "snapshot_nodrug")):
                X = {s: pd.DataFrame(np.column_stack([logit(P[s][snap]), logit(P[s]["baseline2"]), el[s][[CUR[e], *phase]]]),
                                     columns=GROUPS[:3] + phase).assign(subjectid=el[s].subjectid.to_numpy(), **{y: el[s][y].to_numpy()})
                     for s in out}
                cols = GROUPS[:3] + phase
                raw = {}
                raw["train"], _, stk = crossfit(fit_on(lr, cols), lambda mod, d: mod.predict_proba(d[cols])[:, 1], X["train"], folds)
                for s in splits:
                    raw[s] = stk.predict_proba(X[s][cols])[:, 1]
                # isotonic is fit on VAL (spec), so val metrics of the calibrated fusion are calibration-in-sample; test is clean
                lab = X["val"][y].notna().to_numpy()
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw["val"][lab], X["val"][y][lab])
                for s in out:
                    P[s][m] = iso.predict(raw[s])
                keep[m], keep[f"iso_{m}"] = stk, iso
                if m == "fusion" and h == HEAD:
                    for s in splits:  # largest signed contribution coef * standardized input, phase one-hots summed
                        z = stk[:-1].transform(X[s][cols]) * stk[-1].coef_[0]
                        g = np.column_stack([z[:, :3], z[:, 3:].sum(1)])
                        out[s].loc[el[s].index, f"lead_specialist_{e}"] = np.array(GROUPS)[g.argmax(1)]
            saved["fusion"]["models"][f"{e}_{h}"] = {k: keep[k] for k in keep if not k.startswith("snapshot")}

            for s in out:
                for m in MODELS:
                    out[s].loc[el[s].index, f"p_{m}_{e}_{h}"] = P[s][m]
            v = el["val"][y].notna().to_numpy()
            yv = el["val"][y].to_numpy()[v]
            print(f"val {e} {h:>2} min (n={v.sum()}, pos={int(yv.sum())}): " + "  ".join(
                f"{m} {roc_auc_score(yv, P['val'][m][v]):.3f}/{average_precision_score(yv, P['val'][m][v]):.3f}" for m in MODELS),
                flush=True)
    for s in out:  # columns created row-subset-wise; make every p_ column exist and be float even if an event had no rows
        for c in [f"p_{m}_{e}_{h}" for m in MODELS for e in EVENTS for h in HORIZONS]:
            out[s][c] = out[s].get(c, np.nan)
            out[s][c] = out[s][c].astype(float)
    thr = {e: {"threshold": choose_threshold(out["val"], f"p_fusion_{e}_{HEAD}"),
               "alarms_per_hour_target": CFG["alerts"]["alarms_per_hour"], "chosen_on": "val"} for e in EVENTS}
    return out, saved, thr


def main(test=False):
    from orew.features import DRUG_COLS, FEATURES

    if test:
        if LOCK.exists():
            sys.exit(f"refusing --test: {LOCK} exists, the test split was already used")
        if subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode:
            sys.exit("refusing --test: pytest -q failed")
    t0 = time.time()
    pt = pd.read_parquet(PROCESSED / "pred_table.parquet")
    ft = pd.read_parquet(PROCESSED / "features.parquet", columns=["caseid", "t_sec", *[c for c in FEATURES if c not in pt]])
    df = pt.merge(ft, on=["caseid", "t_sec"], how="left", validate="1:1")
    assert len(df) == len(pt)
    folds = {int(k): v for k, v in json.loads((PROCESSED / "folds.json").read_text()).items()}
    print(f"{len(df)} rows, {len(FEATURES)} features ({len(DRUG_COLS)} drug), loaded in {time.time() - t0:.0f} s", flush=True)
    out, saved, thr = run(df, folds, FEATURES, DRUG_COLS, ("val", "test") if test else ("val",))
    OUT.mkdir(parents=True, exist_ok=True)
    for k, mod in saved["snapshot"].items():
        mod.save_model(OUT / f"{k}.json")
    (OUT / "fusion.pkl").write_bytes(pickle.dumps(saved["fusion"]))
    (OUT / "threshold.json").write_text(json.dumps(thr, indent=1))
    out["train"].to_parquet(PROCESSED / "preds_oof.parquet", index=False)
    out["val"].to_parquet(PROCESSED / "preds_val.parquet", index=False)
    if test:
        LOCK.write_text(f"preds_test written by orew.models at {time.ctime()}\n")
        out["test"].to_parquet(PROCESSED / "preds_test.parquet", index=False)
    print("thresholds", {e: round(v["threshold"], 4) for e, v in thr.items()}, f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="also write preds_test.parquet (once; locks reports/TEST_LOCK)")
    main(ap.parse_args().test)
