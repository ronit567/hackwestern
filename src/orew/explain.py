"""Plain-language drivers: TreeSHAP on the headline snapshot models -> SPEC Output records.

`python -m orew.explain --split val|test` writes data/processed/records_{split}.jsonl, one record per
pred_table row of that split, with the fusion probabilities, lead_specialist and the top 3 features
whose SHAP value pushes the 10-min risk up (skipping any that would quote an implausible value).
"""
import argparse
import json
import math
import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb

from orew import CFG, PROCESSED

EVENTS = ["hypotension", "hypoxemia"]  # CFG["events"] also holds max_artifact_frac
HORIZONS = CFG["windows"]["horizons_min"]
HEAD = CFG["windows"]["headline_horizon_min"]
K = 3

SIGNALS = {  # feature prefix -> (plain label, unit)
    "map": ("MAP", "mmHg"), "sbp": ("systolic BP", "mmHg"), "dbp": ("diastolic BP", "mmHg"),
    "nibp_map": ("cuff MAP", "mmHg"), "hr": ("heart rate", "bpm"), "spo2": ("SpO2", "%"),
    "etco2": ("ETCO2", "mmHg"), "rr": ("resp. rate", "/min"), "fio2": ("FiO2", "%"),
    "tv": ("tidal volume", "mL"), "peep": ("PEEP", "mbar"), "pip": ("peak airway pressure", "mbar"),
    "sevo": ("sevoflurane", "%"), "des": ("desflurane", "%"), "mac": ("MAC", ""),
    "ppf_rate": ("propofol rate", "mL/h"), "rftn_ug_h": ("remifentanil rate", "ug/h"),
    "phen_rate": ("phenylephrine rate", "mL/h"), "nepi_rate": ("norepinephrine rate", "mL/h"),
    "bis": ("BIS", ""), "bt": ("temperature", "°C"),
    "beat_pp": ("pulse pressure", "mmHg"), "beat_dpdt": ("max dP/dt", "mmHg/s"),
    "beat_sys_area": ("systolic area", "mmHg·s"), "beat_hr": ("pulse rate", "bpm"),
    "beat_ppv": ("pulse-pressure swing", ""), "beat_sbp_sd": ("beat-to-beat SBP variability", "mmHg"),
    "pleth_amp": ("pleth amplitude", ""), "pleth_cv": ("pleth instability (CV)", ""),
    "shock_index": ("shock index", ""), "baseline_map": ("baseline MAP", "mmHg"),
}
# a quoted level outside these is an artifact (flush, zeroed line, loose probe; e.g. diastolic -30 mmHg, HR 0), so that
# driver is skipped and the next one used. MAP below 30 or a core temperature below 30 °C is not a monitor reading to cite.
PLAUSIBLE = {
    "map": (30, 200), "sbp": (40, 260), "dbp": (15, 160), "nibp_map": (30, 200), "baseline_map": (40, 180),
    "hr": (25, 220), "beat_hr": (25, 220), "spo2": (50, 100), "bt": (30, 43), "beat_pp": (10, 150),
    "beat_sys_area": (0, 50),
}
PHASES = {"pre": "pre-induction", "induction": "induction", "maintenance": "maintenance",
          "emergence": "emergence", "post": "post-anaesthesia"}  # keys = orew.features.PHASES, in order
MARKERS = {"anestart": "anaesthesia start", "phase_start": "this phase started"}
SIG = "|".join(sorted(map(re.escape, SIGNALS), key=len, reverse=True))  # longest prefix wins


def num(v):
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.2g}"


def u(unit):
    return f" {unit}" if unit and unit != "%" else unit


def sig(m):
    return SIGNALS[m["sig"]]


def slope(m, v):
    lab, unit = sig(m)
    return f"{lab} {'falling' if v < 0 else 'rising'} {num(abs(v))}{u(unit)}/min (last {m['w']} min)"


def delta(m, v):
    lab, unit = sig(m)
    if v == 0:
        return f"{lab} unchanged over {m['w']} min"
    return f"{lab} just {'increased' if v > 0 else 'decreased'} ({'+' if v > 0 else '-'}{num(abs(v))}{u(unit)} in {m['w']} min)"


def stat(word):
    return lambda m, v: f"{sig(m)[0]} {word} {num(v)}{u(sig(m)[1])} over last {m['w']} min"


def level(f):
    """f for a quoted level (current, mean or min), or None where that level is implausible for the signal."""
    def g(m, v):
        lo, hi = PLAUSIBLE.get(m["sig"], (-math.inf, math.inf))
        return f(m, v) if lo <= v <= hi else None
    return g


def value(lab, unit=""):
    return lambda m, v: f"{lab} {num(v)}{u(unit)}"


def yes_no(yes, no):
    return lambda m, v: yes if v >= 0.5 else no


# feature-name regex -> formatter(match, current value); first full match wins. Grammar of orew.features.FEATURES.
TEMPLATES = {
    rf"(?P<sig>{SIG})_slope_(?P<w>\d+)m": slope,
    rf"(?P<sig>{SIG})_delta_(?P<w>\d+)m": delta,
    rf"(?P<sig>{SIG})_mean_(?P<w>\d+)m": level(stat("averaging")),
    rf"(?P<sig>{SIG})_min_(?P<w>\d+)m": level(stat("dipped to")),
    rf"(?P<sig>{SIG})": level(lambda m, v: f"{sig(m)[0]} {num(v)}{u(sig(m)[1])}"),
    rf"(?P<sig>{SIG})_missing_frac": lambda m, v: f"{sig(m)[0]} missing {v:.0%} of last 10 min",
    rf"art_(?P<sig>{SIG})_frac": lambda m, v: f"{sig(m)[0]} artifact {v:.0%} of last 10 min",
    r"pleth_amp_rel": lambda m, v: f"pleth amplitude {num(v)}x its 10-min mean",
    r"phase": lambda m, v: f"{list(PHASES.values())[int(v)]} phase",
    r"phase_(?P<p>\w+)": lambda m, v: f"{'' if v >= 0.5 else 'not in '}{PHASES.get(m['p'], m['p'])} phase",
    r"min_since_(?P<mk>\w+)": lambda m, v: f"{num(v)} min since {MARKERS.get(m['mk'], m['mk'].replace('_', ' '))}",
    r"age": value("age", "y"), r"asa": value("ASA"), r"bmi": value("BMI"),
    r"height": value("height", "cm"), r"weight": value("weight", "kg"),
    r"emop": yes_no("emergency operation", "planned operation"), r"sex": yes_no("male", "female"),
    r"department": value("department code"),
    r"preop_htn": yes_no("history of hypertension", "no hypertension history"),
    r"preop_dm": yes_no("diabetic", "not diabetic"),
    r"preop_hb": value("preop Hb", "g/dL"), r"preop_sao2": value("preop SaO2", "%"),
    r"preop_(?P<x>\w+)": lambda m, v: f"preop {m['x'].upper()} {num(v)}",
}
_COMPILED = [(re.compile(k), f) for k, f in TEMPLATES.items()]


@lru_cache(maxsize=None)
def _template(name):
    for rx, f in _COMPILED:
        if m := rx.fullmatch(name):
            return m, f
    return None, None


def describe(name, v):
    """Plain-language string for feature `name` at its current value `v`; None if the value it quotes is implausible."""
    v = math.nan if v is None else float(v)  # float() first: numpy float32 NaN is not a Python float
    m, f = _template(name)
    if math.isnan(v):
        lab = SIGNALS[m["sig"]][0] if m and "sig" in m.re.groupindex else name.replace("_", " ")
        return f"{lab}: no recent data"
    return f(m, v) if f else f"{name.replace('_', ' ')} {num(v)}"


def top_drivers(booster, X):
    """Per row: feature indices by descending SHAP, and whether each is > 0 (pushes risk up)."""
    phi = booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]  # exact TreeSHAP; last column = bias
    order = np.argsort(-phi, axis=1, kind="stable")
    return order, np.take_along_axis(phi, order, 1) > 0


def _p(x):
    return None if x is None or not np.isfinite(x) else float(f"{x:.4g}")


def records(split, processed=PROCESSED):
    preds = pd.read_parquet(processed / f"preds_{split}.parquet")
    boosters = {}
    for e in EVENTS:
        boosters[e] = xgb.Booster()
        boosters[e].load_model(processed / "models" / f"snapshot_{e}_{HEAD}.json")
    cols = sorted({c for b in boosters.values() for c in b.feature_names})
    feat = pd.read_parquet(processed / "features.parquet", columns=["caseid", "t_sec", *cols],
                           filters=[("caseid", "in", preds.caseid.unique().tolist())])
    feat = preds[["caseid", "t_sec"]].merge(feat, on=["caseid", "t_sec"], how="inner", validate="1:1")
    assert len(feat) == len(preds), "prediction rows without a features row: re-run orew.features and orew.models"

    out = {}
    t0 = time.perf_counter()
    for e, b in boosters.items():
        ok = preds[f"eligible_{e}"].to_numpy(bool)
        X = feat.loc[ok, b.feature_names]  # the order the model was trained with
        order, pos = top_drivers(b, X)
        names, vals = np.array(b.feature_names), X.to_numpy()
        drv, txt = [None] * len(preds), [None] * len(preds)
        for j, i in enumerate(np.flatnonzero(ok)):
            drv[i], txt[i] = [], []
            for c in order[j][pos[j]]:  # risk-raising features by SHAP; one quoting an implausible value is skipped
                if (t := describe(names[c], vals[j, c])) is not None:
                    drv[i].append(str(names[c]))
                    txt[i].append(t)
                    if len(drv[i]) == K:
                        break
        out[e] = drv, txt
    secs = time.perf_counter() - t0

    recs = []
    for i, r in enumerate(preds.itertuples(index=False)):
        rec = {"caseid": int(r.caseid), "t_sec": int(r.t_sec)}
        for e in EVENTS:
            rec[e] = None if not getattr(r, f"eligible_{e}") else {
                **{f"p{h}": _p(getattr(r, f"p_fusion_{e}_{h}")) for h in HORIZONS},
                "lead_specialist": getattr(r, f"lead_specialist_{e}"),
                "top_drivers": out[e][0][i], "top_drivers_text": out[e][1][i]}
        recs.append(rec)
    return recs, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    t0 = time.perf_counter()
    recs, secs = records(a.split)
    path = PROCESSED / f"records_{a.split}.jsonl"
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    print(f"{len(recs)} records -> {path}; SHAP+text {secs:.1f} s, total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
