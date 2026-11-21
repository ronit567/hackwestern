"""MADE-UP hypoxemia results, for showing what the results page would look like with more events.

Not model output: no data, model or evaluation goes in. Writes web/illustrative_hypoxemia.json, which the
page loads only with ?illustrative=1 and labels "ILLUSTRATIVE — NOT A REAL RESULT" everywhere. Nothing in
the pipeline reads it. Each block has the shape of metrics.json's hypoxemia entry (PLAN.md "Results frontend").

    uv run python -m orew.illustrative
"""

import json
import math

from orew import ROOT

OUT = ROOT / "web" / "illustrative_hypoxemia.json"
NOTE = ("These numbers are made up for illustration. They show what hypoxemia results could look like with "
        "more surgeries and are not model results: no data, model or evaluation produced them.")
MODELS = ["baseline1", "baseline2", "snapshot", "fusion"]
SKILL = {"baseline1": 0.0, "baseline2": 0.15, "snapshot": 0.7, "fusion": 1.0}  # share of the baseline1 -> fusion gap
HS = ["5", "10", "15"]
HEADLINE = "10"
N = {"cases": 440, "subjects": 431, "times": 182_000, "events": 58}
POS = {"5": 430, "10": 870, "15": 1290}  # times with an onset ahead (~15 per event at 10 min, as in the real val split)
# (baseline1, fusion) point estimates; every other model sits between them by SKILL
AUROC = {"5": (0.75, 0.86), "10": (0.70, 0.83), "15": (0.66, 0.79)}
# AUPRC as a multiple of the prevalence of whatever times it is computed on, so it never falls below chance
LIFT = {"5": (7.5, 20.0), "10": (5.2, 13.0), "15": (4.2, 10.0)}
BAND_LIFT = (1.3, 2.8)  # inside a narrow value band the current number says little, so lifts are small
SENS = {"5": (0.42, 0.62), "10": (0.36, 0.55), "15": (0.31, 0.49)}
# p10 alert simulation, headline horizon only (as in metrics.json): (baseline1, fusion, CI half-width).
# 2 alarms/h over ~1,500 h is ~3,000 alarms for ~40 warned events (~2 true alarms each), hence ~35-50 false per true
ALERT = {
    "false_per_true_alarm": (50.0, 34.0, None), "median_lead_min": (3.1, 5.6, 1.4),
    "pct_warned_5min": (0.16, 0.36, 0.11), "event_recall": (0.45, 0.66, 0.12),
}
THRESHOLD = {"baseline1": 0.013, "baseline2": 0.014, "snapshot": 0.019, "fusion": 0.021}
BINS = [("90-94", 90, 95, 1450, 70, (0.60, 0.73)), (">=95", 95, 101, 179_500, 600, (0.62, 0.77))]  # 90-94: ~5 % positive, as in the real split
# (label, times, share of positives, AUROC offset): times sum to N["times"] in every attribute
GROUPS = {
    "age_band": [("18-39", 21_000, 0.06, 0.02), ("40-59", 62_000, 0.30, 0.0), ("60-74", 78_000, 0.47, -0.01), (">=75", 21_000, 0.17, -0.03)],
    "sex": [("F", 81_000, 0.38, 0.01), ("M", 101_000, 0.62, -0.01)],
    "asa": [("1", 37_000, 0.12, 0.03), ("2", 111_000, 0.55, 0.01), ("3", 26_500, 0.28, -0.02), ("4", 1_500, 0.03, -0.06), ("unknown", 6_000, 0.02, 0.0)],
    "department": [("General surgery", 122_000, 0.55, 0.0), ("Thoracic surgery", 48_000, 0.38, -0.02), ("Gynecology", 6_500, 0.04, 0.04), ("Urology", 5_500, 0.03, 0.02)],
    "emergency": [("emergency", 28_000, 0.2, -0.03), ("planned", 154_000, 0.8, 0.01)],
}
# reliability: decile mean predicted risk relative to prevalence, and how far observed strays from it
SHAPE_B1 = [0.45, 0.5, 0.55, 0.62, 0.7, 0.8, 0.95, 1.2, 1.6, 2.63]
SHAPE_F = [0.09, 0.14, 0.2, 0.27, 0.37, 0.5, 0.7, 1.05, 1.75, 4.93]
WIGGLE_F = [0.1, -0.1, 0.05, -0.05, 0.1, -0.08, 0.06, -0.04, 0.03, -0.01]
WIGGLE_B1 = [-0.5, -0.3, -0.2, 0.1, -0.1, 0.2, 0.0, 0.3, 0.1, 0.05]


def mix(pair, m):
    return pair[0] + SKILL[m] * (pair[1] - pair[0])


def ci(e, half, lo=0.0, hi=1.0):
    return [round(e, 4), round(max(lo, e - half), 4), round(min(hi, e + half), 4)]


def ratio_ci(e, w):  # skewed interval for small positive quantities (AUPRC, Brier, false per true)
    return [round(e, 5), round(e / (1 + w), 5), round(e * (1 + w), 5)]


def events_in(n_pos, H):
    return max(1.0, n_pos * N["events"] / POS[H])


def disc(m, auroc_pair, lift_pair, prev, events, shift=0.0):
    a = mix(auroc_pair, m) + shift
    return {"auroc": ci(a, 0.45 / math.sqrt(events), hi=0.99),
            "auprc": ratio_ci(mix(lift_pair, m) * prev, 2.5 / math.sqrt(events))}


def overall(excl):
    out = {}
    for H in HS:
        prev = POS[H] / N["times"]
        out[H] = {}
        for m in MODELS:
            # flagged times (just after an event) are easy positives, so leaving them out costs a little
            d = disc(m, AUROC[H], LIFT[H], prev * (0.85 if excl else 1.0), N["events"], -0.015 if excl else 0.0)
            d["brier"] = ratio_ci(prev * (1 - prev) * (1 - 0.08 * SKILL[m]), 0.15)
            d["sens_at_90spec"] = ci(mix(SENS[H], m) - (0.02 if excl else 0.0), 0.9 / math.sqrt(N["events"]))
            if H == HEADLINE:
                d["alarms_per_hour"] = [2.0, 1.82, 2.17]  # every threshold is set for 2 alarms per hour
                for k, (b, f, half) in ALERT.items():
                    e = mix((b, f), m)
                    d[k] = ratio_ci(e, 0.45) if half is None else ci(e, half, hi=10.0 if k == "median_lead_min" else 1.0)  # lead <= W
            out[H][m] = d
    return out


def bins(excl):
    out = {}
    for H in HS:
        out[H] = []
        for label, lo, hi, n, n_pos, pair in BINS:
            bn, bp = round(n * (0.97 if excl else 1)), round(n_pos * POS[H] / POS[HEADLINE] * (0.85 if excl else 1))
            band = (pair[0] + AUROC[H][0] - AUROC[HEADLINE][0], pair[1] + AUROC[H][1] - AUROC[HEADLINE][1])
            out[H].append({"label": label, "lo": lo, "hi": hi, "n": bn, "n_pos": bp,
                           "models": {m: disc(m, band, BAND_LIFT, bp / bn, events_in(bp, H), -0.015 if excl else 0.0) for m in MODELS}})
    return out


def subgroups():
    out = {}
    for H in HS:
        out[H] = {}
        for attr, groups in GROUPS.items():
            rows, left = [], POS[H]
            for i, (label, n, share, off) in enumerate(groups):
                n_pos = left if i == len(groups) - 1 else round(share * POS[H])  # positives add up exactly
                left -= n_pos
                prev = (POS[H] / N["times"]) ** 0.2 * (n_pos / n) ** 0.8  # lift shrinks a little where positives are dense
                rows.append({"label": label, "n": n, "n_pos": n_pos,
                             "models": {m: disc(m, AUROC[H], LIFT[H], prev, events_in(n_pos, H), off) for m in MODELS}})
            out[H][attr] = rows
    return out


def reliability():
    out = {}
    for H in HS:
        prev, n = POS[H] / N["times"], N["times"] // 10
        out[H] = {}
        for m in MODELS:
            k = SKILL[m]
            p = [prev * (b + k * (f - b)) for b, f in zip(SHAPE_B1, SHAPE_F)]
            obs = [pi * (1 + wf + (1 - k) * wb) for pi, wf, wb in zip(p, WIGGLE_F, WIGGLE_B1)]
            fix = POS[H] / (n * sum(obs))  # observed positives add up to POS
            out[H][m] = [{"p": round(pi, 6), "obs": round(o * fix, 6), "n": n} for pi, o in zip(p, obs)]
    return out


def build():
    o, oe = overall(False), overall(True)
    return {
        "illustrative": True,
        "note": NOTE,
        "event": "hypoxemia",
        "models": MODELS, "final_model": "fusion", "horizons_min": [int(H) for H in HS],
        "headline_horizon_min": int(HEADLINE), "bootstrap": 1000,
        "n": N,
        "alert_threshold": THRESHOLD,
        "overall": o, "overall_excl_flagged": oe,
        "bins": bins(False), "bins_excl_flagged": bins(True),
        "subgroups": subgroups(),
        "reliability": reliability(),
        "ablation": {H: {"model": "snapshot", "with_drugs": {k: o[H]["snapshot"][k] for k in ("auroc", "auprc")},
                         "without_drugs": disc("snapshot", AUROC[H], LIFT[H], 0.93 * POS[H] / N["times"], N["events"], -0.012)} for H in HS},
    }


def main():
    OUT.write_text(json.dumps(build(), indent=1, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} (made-up numbers, shown only with ?illustrative=1)")


if __name__ == "__main__":
    main()
