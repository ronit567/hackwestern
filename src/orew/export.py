"""Write reports/web/{cohort.json, cases/*.json} for the results page in web/. metrics.json is evaluate.py's job.

Contract: PLAN.md, section "Results frontend".
"""
import argparse
import json

import pandas as pd

from orew import RAW, REPORTS

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


def case_json(df, row):
    df = df.assign(rftn_ug_h=rftn_ug_h(df))
    g = df.groupby(-(-df.t_sec // STEP) * STEP)[SERIES].last()  # last finite value in (t - 10, t], like the 2 s grid
    t0 = row.casestart
    return {
        "caseid": int(row.name),
        "demographics": {k: val(row[k]) for k in DEMO},
        "duration_min": val(round((row.caseend - t0) / 60, 1)),
        "phases_min": {k: val(round((row[k] - t0) / 60, 2)) for k in PHASES},
        "t_min": ((g.index - t0) / 60).round(2).tolist(),
        "series": {c: g[c].round(1).astype(object).where(g[c].notna(), None).tolist() for c in SERIES},
    }


def export(max_cases=50, out=OUT):
    cases = pd.read_csv(RAW / "cases.csv", encoding="utf-8-sig").set_index("caseid")
    cohort = pd.read_csv(RAW / "cohort.csv")
    ids = [c for c in cohort.caseid if (RAW / f"{c:04d}.parquet").exists()]
    pct_time, has, index = {}, {}, []
    for i, c in enumerate(ids):
        df = pd.read_parquet(RAW / f"{c:04d}.parquet")
        pct_time[c], has[c] = df.drop(columns="t_sec").notna().mean(), df.drop(columns="t_sec").notna().any()
        if i < max_cases:
            case = case_json(df, cases.loc[c])
            dump(case, out / "cases" / f"{c}.json")
            index.append({"caseid": int(c), "duration_min": case["duration_min"], **case["demographics"]})
    dump(index, out / "cases" / "index.json")

    c = cases.loc[ids]
    age = c.age.quantile([0, 0.25, 0.5, 0.75, 1]).round(1).tolist()
    pt = pd.DataFrame(pct_time)  # signals x cases
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
    }, out / "cohort.json")
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, default=50, help="case files to export (cohort.json always uses all cached)")
    ids = export(ap.parse_args().max_cases)
    print(f"{len(ids)} cached cases -> {OUT}")
