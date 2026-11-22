"""Live replay of one unseen test surgery: every 30 s of surgery time the saved models score the cleaned 2 s grid up to
that time only, exactly as orew.models scored preds_test.parquet (served by orew.app; PLAN.md "Live demo")."""
import json
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from orew import PROCESSED, RAW, ROOT
from orew.clean import STEP
from orew.evaluate import HH, HS, WIN, alarms  # the p10 alert simulator evaluate reports with
from orew.evaluate import STEP as TICK
from orew.explain import pick_drivers, top_drivers
from orew.export import DEMO, PHASES, val
from orew.features import case_features, clinical
from orew.models import CUR, SLOPE, lead_specialist, stack_inputs

E = "hypotension"  # hypoxemia is exploratory (8 warnable test events): the demo replays hypotension only
# the cleaned causal grid columns the model reads (artifact -> null), so the MAP tile shows what the table, banner and
# map_now show; raw map_raw read 1, 9, -12 mmHg on 287's zero line before the catheter, drawn as "below 65"
VITALS = {k: k for k in ("map", "hr", "spo2", "sbp", "dbp", "bt", "rr", "etco2")}
VSTEP = 10  # s, vitals resolution (the results page's case files use 10 s too)


def local_grid(caseid):
    """The one data-source seam: a surgery's cleaned causal 2 s grid (labels.py output). LiveModel(grid=...) swaps it."""
    return pd.read_parquet(PROCESSED / "grid" / f"{caseid:04d}.parquet")


def r(x, d=1):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


class LiveModel:
    """Saved snapshot XGBoost + fusion stacker + isotonic + val threshold, and the test surgeries' clinical rows,
    prediction-time eligibility and events. Only the test split: never used for training, calibration or thresholds."""

    def __init__(self, grid=local_grid, data_source="local", processed=PROCESSED):
        self.grid, self.data_source = grid, data_source
        m = processed / "models"
        self.snap = {h: XGBClassifier() for h in HS}
        for h, mod in self.snap.items():
            mod.load_model(m / f"snapshot_{E}_{h}.json")
        fz = pickle.loads((m / "fusion.pkl").read_bytes())
        self.phase, self.fus = fz["phase"], {h: fz["models"][f"{E}_{h}"] for h in HS}
        thr = json.loads((m / "threshold.json").read_text())[E]
        assert thr["chosen_on"] == "val", "alert threshold must come from val (leakage #10)"
        self.thr = float(thr["threshold"])
        cases = pd.read_csv(RAW / "cases.csv", encoding="utf-8-sig")
        self.clin, self.cases = clinical(cases), cases.set_index("caseid")
        pt = pd.read_parquet(processed / "pred_table.parquet", filters=[("split", "==", "test")],
                             columns=["caseid", "t_sec", f"eligible_{E}", f"next_onset_{E}_s"])
        self.pt = {int(c): d.set_index("t_sec") for c, d in pt.groupby("caseid")}
        ev = pd.read_parquet(processed / "events.parquet")
        ev = ev[(ev.event == E) & ev.caseid.isin(self.pt)]
        self.onsets = {int(c): d.onset_sec.sort_values().tolist() for c, d in ev.groupby("caseid")}
        self.ids = sorted(self.pt)

    def first_pred_sec(self, caseid):
        return int(self.pt[caseid].index[0])

    def surgeries(self, featured=ROOT / "web" / "featured.json"):
        """Test surgeries, featured first (featured.json order), then by caseid; n_events = warnable hypotension events."""
        reason = {f["caseid"]: f["reason"] for f in json.loads(featured.read_text())}
        out = [{"caseid": c, "duration_min": r((self.cases.caseend[c] - self.cases.casestart[c]) / 60),
                "n_events": sum(o > self.first_pred_sec(c) for o in self.onsets.get(c, [])),
                "featured": c in reason, "reason": reason.get(c)} for c in self.ids]
        rank = list(reason)
        return sorted(out, key=lambda s: (rank.index(s["caseid"]) if s["featured"] else len(rank), s["caseid"]))

    def score(self, g, t, c):
        """(p5, p10, p15, snapshot feature row, current MAP, headline-horizon parts) at t from grid rows t_sec <= t only,
        built the way orew.models built the fusion inputs; p is the running max over horizons, as in preds_*.parquet.
        parts: the 10-min snapshot and baseline 1 probabilities and lead_specialist, as preds_*.parquet stores them."""
        g = g.iloc[: t // STEP + 1]  # the grid is gap-free from t_sec = 0, so these are exactly the rows <= t
        f = case_features(g, [t], c)
        cur = pd.DataFrame({CUR[E]: g["map"].to_numpy()[-1:]})  # = pred_table map_now (baseline 1's input)
        p = []
        for h in HS:
            m = self.fus[h]
            snap = self.snap[h].predict_proba(f)[:, 1]
            b2 = m["baseline2"].predict_proba(cur.assign(**{SLOPE[E]: f[SLOPE[E]].to_numpy()}))[:, 1]
            x = stack_inputs(snap, b2, cur.assign(**{k: f[k].to_numpy() for k in self.phase}), self.phase)
            p.append(m["iso_fusion"].predict(m["fusion"].predict_proba(x)[:, 1])[0])
            if h == HH:
                parts = {"p_snapshot_10": float(snap[0]), "p_baseline1_10": float(m["baseline1"].predict_proba(cur)[0, 1]),
                         "lead": str(lead_specialist(m["fusion"], x)[0])}
        return (*np.fmax.accumulate(p), f, float(cur.iat[0, 0]), parts)

    def drivers(self, f):
        """Plain-language SHAP drivers of the headline snapshot model, as explain.records writes them."""
        b = self.snap[HH].get_booster()
        x = f[b.feature_names]
        order, pos = top_drivers(b, x)
        return pick_drivers(np.array(b.feature_names), order[0], pos[0], x.to_numpy()[0])[1]

    def replay(self, caseid):
        """{meta, steps, vitals, summary} of one test surgery. steps: one per 30 s of surgery time (t = 0, 30, ...), each
        computed from grid rows <= t only, with the alarm that fired at t (evaluate's rule) and the event onsets in
        (t - 30, t] (ground truth from events.parquet). vitals[i]: t_sec = 10 i. summary: the export's per-surgery
        verdict (detection), known only at the end."""
        g, c, pt = self.grid(caseid), self.clin.loc[caseid], self.pt[caseid]
        first, onsets = self.first_pred_sec(caseid), self.onsets.get(caseid, [])
        steps, hist_p, hist_t, fired, leads = [], [], [], [], []
        for t in range(0, int(g.t_sec.iat[-1]) + 1, TICK):
            # scored = the evaluation's eligibility from pred_table: a drop only becomes an event once MAP has stayed
            # under 65 for 60 s, so like the evaluation (and the risk curves export draws) the replay shows no risk
            # while a drop is in progress or the input window is mostly missing. Probabilities are computed live.
            s = {"t_sec": t, "scored": bool(pt[f"eligible_{E}"].get(t, False)), "p5": None, "p10": None, "p15": None,
                 "p_snapshot_10": None, "p_baseline1_10": None, "lead": None,
                 "map_now": r(g["map"].iat[t // STEP]), "alarm": None, "onsets": []}
            if s["scored"]:
                s["p5"], s["p10"], s["p15"], f, now, parts = self.score(g, t, c)
                s.update(parts)
                hist_p.append(s["p10"])
                hist_t.append(t)
                if alarms(np.array(hist_p), np.zeros(len(hist_t)), np.array(hist_t), self.thr)[-1]:
                    s["alarm"] = {"t_sec": t, "p10": s["p10"], "map_now": r(now), "drivers": self.drivers(f)}
                    fired.append(t)
            for o in (o for o in onsets if t - TICK < o <= t):
                a = next((x for x in fired if o - WIN <= x < o), None) if o > first else None
                lead = None if a is None else round(round(o / 60, 2) - round(a / 60, 2), 2)  # export's minutes, 2 dp
                s["onsets"].append({"t_sec": int(o), "event": E, "warnable": bool(o > first), "warned": a is not None,
                                    "lead_min": lead, "alarm_t_sec": a})
                if a is not None:
                    leads.append(lead)
            steps.append(s)
        v = g.groupby(-(-g.t_sec // VSTEP) * VSTEP)[list(VITALS.values())].last()  # last finite value in (t - 10, t]
        assert (v.index == np.arange(len(v)) * VSTEP).all(), "vitals[i] must be t_sec = 10 i (gap-free grid)"
        vitals = [{"t_sec": int(t), **{k: r(x[col]) for k, col in VITALS.items()}} for t, x in v.iterrows()]
        nxt = pt[f"next_onset_{E}_s"]
        meta = {"caseid": caseid, "duration_s": int(g.t_sec.iat[-1]), "threshold": self.thr,
                "demographics": {k: val(self.cases.at[caseid, k]) for k in DEMO},
                "phases_min": {k: r((self.cases.at[caseid, k] - self.cases.casestart[caseid]) / 60, 2) for k in PHASES},
                "first_pred_sec": first}
        summary = {"events": sum(o > first for o in onsets), "warned": len(leads),
                   "median_lead_min": round(float(np.median(leads)), 2) if leads else None, "alarms": len(fired),
                   "false_alarms": int(sum(not nxt[x] <= WIN for x in fired))}  # NaN next onset = false
        return {"meta": meta, "steps": steps, "vitals": vitals, "summary": summary}
