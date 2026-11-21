"""Write reports/web/{cohort.json, cases/*.json} for the results page in web/. metrics.json is evaluate.py's job.

Contract: PLAN.md, section "Results frontend".
"""
import argparse
import json

import pandas as pd

from orew import PROCESSED, RAW, REPORTS

OUT = REPORTS / "web"
STEP = 10  # s, case-explorer resolution
SERIES = ["map", "nibp_map", "spo2", "hr", "etco2", "bis", "ppf_rate", "rftn_ug_h"]
PHASES = ["anestart", "opstart", "opend", "aneend"]
DEMO = ["age", "sex", "height", "weight", "bmi", "asa", "emop", "department", "optype", "opname"]


def val(v):
    """JSON-safe scalar: NaN -> None, numpy -> python."""
    return None if pd.isna(v) else v.item() if hasattr(v, "item") else v


def counts(s):
    return {str(k): int(n) for k, n in s.value_counts(dropna=False).items()}


def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, allow_nan=False, separators=(",", ":")))


def rftn_ug_h(df):
    """Both remifentanil pumps in ug/h, as clean.py does (20 and 50 ug/mL); older caches lack rftn50_rate."""
    return (20 * df.rftn_rate).add(50 * df.get("rftn50_rate", pd.Series(dtype=float)), fill_value=0)


def labels():
    """labels.py outputs: events table and {subjectid: split}, each None until labels.py has run."""
    ev, sp = PROCESSED / "events.parquet", PROCESSED / "splits.json"
    return (pd.read_parquet(ev) if ev.exists() else None,
            {s: k for k, v in json.loads(sp.read_text()).items() for s in v} if sp.exists() else None)


def case_json(df, row, ev=None, split_of=None):
    df = df.assign(rftn_ug_h=rftn_ug_h(df))
    g = df.groupby(-(-df.t_sec // STEP) * STEP)[SERIES].last()  # last finite value in (t - 10, t], like the 2 s grid
    t0 = row.casestart
    out = {
        "caseid": int(row.name),
        "demographics": {k: val(row[k]) for k in DEMO},
        "duration_min": val(round((row.caseend - t0) / 60, 1)),
        "phases_min": {k: val(round((row[k] - t0) / 60, 2)) for k in PHASES},
        "t_min": ((g.index - t0) / 60).round(2).tolist(),
        "series": {c: g[c].round(1).astype(object).where(g[c].notna(), None).tolist() for c in SERIES},
    }
    if split_of:
        out["split"] = split_of.get(row.subjectid)
    # a labelled case with no events has no rows in events.parquet, so "labelled" = labels.py wrote its grid
    if ev is not None and (PROCESSED / "grid" / f"{row.name:04d}.parquet").exists():
        out["events"] = [{"type": r.event, "onset_min": round((r.onset_sec - t0) / 60, 2), "end_min": round((r.end_sec - t0) / 60, 2)}
                         for r in ev[ev.caseid == row.name].itertuples()]
    return out


def export(max_cases=50, out=OUT):
    cases = pd.read_csv(RAW / "cases.csv", encoding="utf-8-sig").set_index("caseid")
    cohort = pd.read_csv(RAW / "cohort.csv")
    ids = [c for c in cohort.caseid if (RAW / f"{c:04d}.parquet").exists()]
    ev, split_of = labels()
    pct_time, has, index = {}, {}, []
    for i, c in enumerate(ids):
        df = pd.read_parquet(RAW / f"{c:04d}.parquet")
        pct_time[c], has[c] = df.drop(columns="t_sec").notna().mean(), df.drop(columns="t_sec").notna().any()
        if i < max_cases:
            case = case_json(df, cases.loc[c], ev, split_of)
            dump(case, out / "cases" / f"{c}.json")
            index.append({"caseid": int(c), "duration_min": case["duration_min"], **case["demographics"], "split": case.get("split")})
    dump(index, out / "cases" / "index.json")

    c = cases.loc[ids]
    age = c.age.quantile([0, 0.25, 0.5, 0.75, 1]).round(1).tolist()
    pt = pd.DataFrame(pct_time)  # signals x cases
    later = {}
    if ev is not None:
        later["events"] = {e: {"n_events": int(n), "n_cases": int(k)} for e, (n, k) in ev.groupby("event").caseid.agg(["size", "nunique"]).iterrows()}
    if split_of:
        s = cohort.assign(split=cohort.subjectid.map(split_of))  # the split covers the whole cohort, not only cached cases
        later["splits"] = {k: {"n_subjects": int(g.subjectid.nunique()), "n_cases": len(g)} for k, g in s.groupby("split")}
    dump({
        "n_cohort": len(cohort),
        "n_cached": len(ids),
        "n_subjects": int(cohort[cohort.caseid.isin(ids)].subjectid.nunique()),
        "age": dict(zip(["min", "q1", "median", "q3", "max"], map(val, age))),
        "sex": counts(c.sex.fillna("unknown")),
        "asa": counts(c.asa.astype("Int64").astype("string").fillna("unknown")),
        "department": counts(c.department.fillna("unknown")),
        "emergency": {"n": int(c.emop.sum()), "of": len(c)},
        "median_duration_min": val(round(((c.caseend - c.casestart) / 60).median(), 1)),
        "signals": [
            {"name": k, "pct_time": round(t * 100, 1), "pct_cases": round(n * 100, 1)}
            for k, t, n in zip(pt.index, pt.mean(axis=1), pd.DataFrame(has).mean(axis=1))
        ],
        **later,
    }, out / "cohort.json")
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=50, help="case files to export (cohort.json always uses all cached)")
    ids = export(ap.parse_args().max_cases)
    print(f"{len(ids)} cached cases -> {OUT}")
