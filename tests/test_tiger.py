"""Tiger Cloud integration (orew.tiger). All but the last test need no database; the last one runs only when
TIGER_DATABASE_URL is set (environment or .env) and works in a throwaway schema."""
import io
import json
import uuid
from types import SimpleNamespace

import pandas as pd
import pytest

from orew import PROCESSED, REPORTS, tiger
from orew.live import LiveModel, local_grid

pytestmark = pytest.mark.skipif(not (PROCESSED / "grid" / "0287.parquet").exists(), reason="needs the processed data")


@pytest.fixture
def no_db(monkeypatch, tmp_path):
    """No TIGER_DATABASE_URL in the environment, and an empty .env path instead of the repo's."""
    monkeypatch.delenv(tiger.KEY, raising=False)
    monkeypatch.setattr(tiger, "ENV", tmp_path / ".env")
    return tmp_path / ".env"


def test_dotenv_and_url(no_db, monkeypatch):
    assert tiger.dotenv() == {} and tiger.url() is None
    u = "postgres://tsdbadmin:pw@x.tsdb.cloud.timescale.com:31234/tsdb?sslmode=require"
    no_db.write_text(f"# a comment\n\nexport OTHER='a=b'\n{tiger.KEY}=\"{u}\"\nnot a setting\n# {tiger.KEY}=old\n")
    assert tiger.dotenv() == {"OTHER": "a=b", tiger.KEY: u}
    assert tiger.url() == u
    monkeypatch.setenv(tiger.KEY, "postgres://env-wins")
    assert tiger.url() == "postgres://env-wins"
    msg = 'connection to server at "x.tsdb.cloud.timescale.com", port 31234 failed: FATAL: password for "tsdbadmin"'
    monkeypatch.setenv(tiger.KEY, u)
    assert tiger.redact(msg) == 'connection to server at "***", port 31234 failed: FATAL: password for "***"'


def test_schema_sql_matches_the_grid():
    dt = tiger.grid_dtypes()
    assert dt.equals(local_grid(287).dtypes)
    tables, timescale = tiger.schema_sql(dt)
    grid = next(s for s in tables if s.startswith("CREATE TABLE IF NOT EXISTS grid ("))
    cols = grid[grid.index("(") + 1:-1].split(", ")
    assert cols[0] == "caseid integer NOT NULL" and [c.split()[0] for c in cols[1:]] == list(dt.index)
    assert {"t_sec integer NOT NULL", "map double precision", "art_map boolean", "map_missing smallint"} <= set(cols)
    assert all("IF NOT EXISTS" in s for s in tables)
    for h in ("grid", "predictions"):
        assert f"SELECT create_hypertable('{h}', by_range('t_sec', 3600), if_not_exists => TRUE)" in timescale
        assert any(s.startswith(f"ALTER TABLE {h} SET (timescaledb.compress, timescaledb.compress_segmentby = "
                                "'caseid'") for s in timescale)
    assert "time_bucket(60, t_sec)" in timescale[-1] and "timescaledb.continuous" in timescale[-1]


def test_copy_text_is_exact():
    """The COPY csv text loses nothing: NaN -> empty (NULL), bools, int8 flags and every float bit come back."""
    g = local_grid(287)
    back = pd.read_csv(io.StringIO(tiger.csv(g)), header=None, names=list(g.columns), dtype=g.dtypes.to_dict())
    pd.testing.assert_frame_equal(back, g)


def test_frames_match_the_case_files():
    f = tiger.frames([287, 1046])
    p = f["predictions"]
    assert set(p.caseid) == {287, 1046}
    assert all((p[k].notna() == p.scored).all() for k in ("p5", "p10", "p15", "p_snapshot_10", "p_baseline1_10"))
    for c in (287, 1046):
        case = json.loads((REPORTS / "web" / "cases" / f"{c}.json").read_text())
        al = f["alarms"][f["alarms"].caseid == c]
        assert [(int(t), bool(x), round(float(m), 1)) for t, x, m in zip(al.t_sec, al.is_true, al.map_now)] == \
            [(round(a["t_min"] * 60), a["true"], a["now"]) for a in case["alerts"] if a["type"] == "hypotension"]
        ev = f["events"][(f["events"].caseid == c) & (f["events"].event == "hypotension")]
        assert list(ev.onset_sec / 60) == pytest.approx([e["onset_min"] for e in case["events"]
                                                         if e["type"] == "hypotension"], abs=0.005)


def test_grid_source_falls_back_without_a_database(no_db, monkeypatch, capsys):
    src = tiger.GridSource()
    assert src.name == "local"
    pd.testing.assert_frame_equal(src(287), local_grid(287))
    monkeypatch.setenv(tiger.KEY, "postgresql://orew:s3cretpw@127.0.0.1:1/tsdb")  # nothing listens on port 1
    src = tiger.GridSource()
    assert src.name == "local"
    pd.testing.assert_frame_equal(src(287), local_grid(287))
    out = capsys.readouterr()
    assert "Tiger Cloud not used" in out.err and "s3cretpw" not in out.out + out.err
    assert "127.0.0.1" not in out.out + out.err


def test_insights_unavailable_without_a_database(no_db):
    assert tiger.insights() == {"available": False, "reason": "TIGER_DATABASE_URL not set"}
    assert all(q["title"] and q["plain"] and q["sql"] for q in tiger.QUERIES)
    assert "hypoxemia" not in json.dumps(tiger.QUERIES).lower()  # exploratory: no hypoxemia claim
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from orew.app import create_app
    c = TestClient(create_app(model=SimpleNamespace(grid=local_grid, data_source="local")))
    assert c.get("/api/insights").json() == {"available": False, "reason": "TIGER_DATABASE_URL not set"}
    assert c.get("/api/health").json()["data_source"] == "local"
    c = TestClient(create_app(model=SimpleNamespace(grid=SimpleNamespace(name="tiger"), data_source="local")))
    assert c.get("/api/health").json()["data_source"] == "tiger"


@pytest.mark.skipif(not tiger.url(), reason="TIGER_DATABASE_URL not set")
def test_tiger_roundtrip():
    """Load 287 and 1046 into a throwaway schema: the grid comes back equal to the parquet, the live replay from it
    equals the local one, and every insight query runs."""
    schema = f"orew_test_{uuid.uuid4().hex[:8]}"
    with tiger.connect() as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    try:
        tiger.load(schema, [287, 1046])
        src = tiger.GridSource(schema)
        assert src.name == "tiger"
        for c in (287, 1046):
            pd.testing.assert_frame_equal(src(c), local_grid(c))
        m = LiveModel()
        local = m.replay(287)
        m.grid = src
        assert m.replay(287) == local
        out = tiger.insights(schema)
        assert out["available"] and [q["title"] for q in out["queries"]] == [q["title"] for q in tiger.QUERIES]
        assert all(q["rows"] for q in out["queries"])
    finally:
        with tiger.connect() as conn:
            conn.execute(f"DROP MATERIALIZED VIEW IF EXISTS {schema}.vitals_1min CASCADE")
            conn.execute(f"DROP SCHEMA {schema} CASCADE")
