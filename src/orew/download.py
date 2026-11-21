"""Phase 1: cohort selection and per-case cache (uv run python -m orew.download [--n 5])."""
import argparse

import numpy as np
import pandas as pd

from orew import CFG, RAW

API = "https://api.vitaldb.net"


def cohort():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in ("cases", "trks"):
        if not (RAW / f"{name}.csv").exists():
            pd.read_csv(f"{API}/{name}", encoding="utf-8-sig").to_csv(RAW / f"{name}.csv", index=False)
    if (RAW / "cohort.csv").exists():
        return pd.read_csv(RAW / "cohort.csv")
    cases = pd.read_csv(RAW / "cases.csv")
    trks = pd.read_csv(RAW / "trks.csv")
    c = CFG["cohort"]
    has = trks[trks.tname.isin(c["required_tracks"])].groupby("caseid").tname.nunique()
    ok = (
        (pd.to_numeric(cases.age, errors="coerce") >= c["min_age"])
        & (cases.ane_type == c["ane_type"])
        & (cases.caseend - cases.casestart >= c["min_duration_s"])
        & cases.caseid.isin(has[has == len(c["required_tracks"])].index)
    )
    elig = cases[ok].sort_values("caseid")
    pick = np.random.default_rng(CFG["seed"]).choice(len(elig), CFG["n_cases"], replace=False)
    out = elig.iloc[pick][["caseid", "subjectid"]]
    out.to_csv(RAW / "cohort.csv", index=False)
    print(f"{len(elig)} eligible cases, sampled {len(out)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    args = ap.parse_args()
    print(cohort().head(args.n))
