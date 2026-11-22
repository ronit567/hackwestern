"""The live demo API (PLAN.md "Live demo"): test surgeries only, the SSE replay, and what is (not) served."""
import json

import pandas as pd
import pytest

from orew import PROCESSED

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from orew.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    if not (PROCESSED / "models" / "fusion.pkl").exists():
        pytest.skip("needs the saved models")
    return TestClient(create_app())


def stream(client, url, stop=None):
    """[(event, data)] of an SSE response, closed early once stop(event, data) is true."""
    out, ev = [], None
    with client.stream("GET", url) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"
        for line in r.iter_lines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                out.append((ev, json.loads(line[6:])))
                if stop and stop(*out[-1]):
                    break
    return out


def test_health_and_surgeries(client):
    assert client.get("/api/health").json() == {"ok": True, "models": True, "data_source": "local"}
    s = client.get("/api/surgeries").json()["surgeries"]
    pt = pd.read_parquet(PROCESSED / "pred_table.parquet", columns=["caseid", "split"]).drop_duplicates()
    assert {x["caseid"] for x in s} == set(pt.caseid[pt.split == "test"])
    assert [x["caseid"] for x in s[:4]] == [287, 5566, 3449, 1046] and all(x["reason"] for x in s[:4])
    assert not any(x["featured"] for x in s[4:])


def test_replay_rejects_anything_but_a_test_caseid(client):
    pt = pd.read_parquet(PROCESSED / "pred_table.parquet", columns=["caseid", "split"]).drop_duplicates()
    train = int(pt.caseid[pt.split == "train"].iloc[0])
    for bad in [train, "abc", "287.5", "-287", "%2e%2e"]:
        assert client.get(f"/api/replay/{bad}").status_code == 404, bad


def test_replay_stream(client):
    """287 from 25 min at the top speed: 15 min of vitals first, the alarm at 31 min, then both onsets warned."""
    got = stream(client, "/api/replay/287?speed=100000&start_sec=1500", lambda e, d: e == "onset" and d["t_sec"] == 2346)
    kinds = [e for e, _ in got]
    assert kinds[0] == "meta" and {"tick", "alarm", "onset"} <= set(kinds)
    meta, ticks = got[0][1], [d for e, d in got if e == "tick"]
    assert meta["speed"] == 600 and meta["caseid"] == 287 and meta["start_sec"] == 1500 and meta["threshold"] > 0
    assert ticks[0]["t_sec"] == 1500 and len(ticks[0]["vitals"]) == 90 and ticks[0]["vitals"][-1]["t_sec"] == 1500
    assert all(len(t["vitals"]) == 3 for t in ticks[1:])
    assert all(v["t_sec"] <= t["t_sec"] for t in ticks for v in t["vitals"])  # the extra vitals are causal too
    assert set(ticks[0]["vitals"][0]) == {"t_sec", "map", "hr", "spo2", "sbp", "dbp", "bt", "rr", "etco2"}
    sc = [t for t in ticks if t["scored"]]
    assert sc and all(t["lead"] in ("snapshot", "trend", "current_value", "phase") and t["p_snapshot_10"] is not None
                      and t["p_baseline1_10"] is not None for t in sc)
    assert all(t["lead"] is None and t["p_snapshot_10"] is None for t in ticks if not t["scored"])
    assert [t["t_sec"] for t in ticks] == list(range(1500, ticks[-1]["t_sec"] + 1, 30))
    alarm = next(d for e, d in got if e == "alarm")
    assert alarm["t_sec"] == 1860 and len(alarm["drivers"]) == 3
    onsets = [d for e, d in got if e == "onset"]
    assert [(o["t_sec"], o["warned"], o["alarm_t_sec"]) for o in onsets] == [(2128, True, 1860), (2346, True, 1860)]


def test_replay_ends_with_summary(client):
    got = stream(client, "/api/replay/287?speed=600&start_sec=9300")
    assert got[-1] == ("end", {"summary": {"events": 2, "warned": 2, "median_lead_min": 6.29, "alarms": 2,
                                           "false_alarms": 0}})
    assert got[1][0] == "tick" and got[1][1]["t_sec"] == 9300
    # a seek past the last tick (duration 9544 s, last tick 9540 s) still gets that tick and its 15 min of vitals
    got = stream(client, "/api/replay/287?speed=600&start_sec=99999999999999999999999")
    ticks = [d for e, d in got if e == "tick"]
    assert got[0][1]["start_sec"] == 9540 and [t["t_sec"] for t in ticks] == [9540] and len(ticks[0]["vitals"]) == 90


def test_static_layout(client, monkeypatch):
    assert client.get("/web/index.html").status_code == 200
    assert client.get("/web/").status_code == 200
    assert client.get("/reports/web/metrics.json").json()["split"] == "test"
    for p in ["/data/processed/splits.json", "/README.md", "/src/orew/app.py", "/.env", "/web/%2e%2e/README.md",
              "/reports/web/%2e%2e/test.lock"]:
        assert client.get(p).status_code == 404, p
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/web/live.html"
    monkeypatch.delenv("TIGER_DATABASE_URL", raising=False)
    assert client.get("/api/insights").json() == {"available": False, "reason": "TIGER_DATABASE_URL not set"}
