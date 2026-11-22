"""Tiger Cloud (managed TimescaleDB) for the live demo: the unseen test surgeries' 2 s grid, predictions, events and
alarms in Postgres, the live engine's grid source, and the showcase queries behind /api/insights (README "Tiger Data").

uv run python -m orew.tiger load     # (re)load the 140 test surgeries; needs TIGER_DATABASE_URL (environment or .env)
uv run python -m orew.tiger check    # row counts per table, then every insight query

SQL targets TimescaleDB 2.13+ (create_hypertable with by_range). The compression calls use the 2.x names, which 2.18+
also calls the columnstore (enable_columnstore, convert_to_columnstore, hypertable_columnstore_stats).
"""
import argparse
import json
import os
import sys
from decimal import Decimal

import pandas as pd
import psycopg
from psycopg import sql

from orew import CFG, PROCESSED, ROOT
from orew.evaluate import HS, WIN, alarms
from orew.live import E, local_grid

KEY, ENV = "TIGER_DATABASE_URL", ROOT / ".env"
CHUNK = 3600  # s per chunk; t_sec restarts at 0 in every surgery, so chunk k holds hour k of every surgery
BATCH = 10  # surgeries per grid COPY
SQLTYPE = {"int64": "integer", "float64": "double precision", "bool": "boolean", "int8": "smallint"}
LOW, AHEAD = CFG["events"][E]["threshold"], max(HS) * 60  # 65 mmHg; 15 min


def dotenv(path=None):
    """KEY=VALUE lines of the gitignored .env (`export ` prefix, quotes and # comment lines allowed). Never printed."""
    path, out = path or ENV, {}
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().removeprefix("export ").partition("=")
            if sep and k.strip() and not k.startswith("#"):
                out[k.strip()] = v.strip().strip("\"'")
    return out


def url():
    """The connection string: the environment wins over .env; None when neither has one."""
    return os.environ.get(KEY) or dotenv().get(KEY) or None


def redact(msg):
    """msg with the connection string's host, user and password blanked (libpq errors quote the host)."""
    try:
        parts = psycopg.conninfo.conninfo_to_dict(url() or "")
    except psycopg.Error:
        return "(details hidden)"
    for k in ("password", "host", "user"):
        if parts.get(k):
            msg = msg.replace(str(parts[k]), "***")
    return msg


def warn(what, e):
    print(f"tiger: {what} ({type(e).__name__}: {' '.join(redact(str(e)).split())})", file=sys.stderr)


def connect(schema=None):
    """Autocommit (refresh_continuous_aggregate cannot run inside a transaction); `schema` goes first on the path."""
    if not url():
        raise RuntimeError(f"{KEY} not set (environment or the gitignored .env)")
    conn = psycopg.connect(url(), autocommit=True, connect_timeout=10)
    if schema:
        conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
    return conn


def grid_dtypes():
    """Columns and dtypes of the cleaned 2 s grid (labels.py writes every surgery's with the same schema)."""
    return pd.read_parquet(min((PROCESSED / "grid").glob("*.parquet"))).dtypes


def schema_sql(dtypes):
    """(tables, timescale): idempotent DDL. load() runs TRUNCATE between the two, because changing the compression
    settings fails while compressed chunks exist."""
    grid = ", ".join(f"{c} {SQLTYPE[str(t)]}{' NOT NULL' if c == 't_sec' else ''}" for c, t in dtypes.items())
    tables = [
        "CREATE EXTENSION IF NOT EXISTS timescaledb",
        f"CREATE TABLE IF NOT EXISTS grid (caseid integer NOT NULL, {grid})",
        "CREATE TABLE IF NOT EXISTS predictions (caseid integer NOT NULL, t_sec integer NOT NULL, p5 double precision, "
        "p10 double precision, p15 double precision, scored boolean NOT NULL, map_now double precision, "
        "p_snapshot_10 double precision, p_baseline1_10 double precision)",
        # a predictions table from before the 10-min specialist columns existed (nullable, so fine on compressed chunks)
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS p_snapshot_10 double precision",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS p_baseline1_10 double precision",
        "CREATE TABLE IF NOT EXISTS events (caseid integer NOT NULL, event text NOT NULL, onset_sec integer NOT NULL, "
        "end_sec integer NOT NULL, duration_s integer NOT NULL)",
        "CREATE TABLE IF NOT EXISTS alarms (caseid integer NOT NULL, t_sec integer NOT NULL, "
        "p10 double precision NOT NULL, map_now double precision, is_true boolean NOT NULL)",
        "CREATE INDEX IF NOT EXISTS events_caseid_onset ON events (caseid, onset_sec)",
    ]
    timescale = []
    for h in ("grid", "predictions"):
        timescale += [
            f"SELECT create_hypertable('{h}', by_range('t_sec', {CHUNK}), if_not_exists => TRUE)",
            f"CREATE UNIQUE INDEX IF NOT EXISTS {h}_caseid_t ON {h} (caseid, t_sec)",
            f"ALTER TABLE {h} SET (timescaledb.compress, timescaledb.compress_segmentby = 'caseid', "
            "timescaledb.compress_orderby = 't_sec')",
        ]
    timescale += [
        # a continuous aggregate on an integer time column needs an integer "now"; no refresh policy, load refreshes it
        "CREATE OR REPLACE FUNCTION grid_now() RETURNS integer LANGUAGE sql STABLE SET search_path FROM CURRENT "
        "AS $$ SELECT coalesce(max(t_sec), 0) FROM grid $$",
        "SELECT set_integer_now_func('grid', 'grid_now', replace_if_exists => TRUE)",
        "CREATE MATERIALIZED VIEW IF NOT EXISTS vitals_1min WITH (timescaledb.continuous) AS "
        "SELECT caseid, time_bucket(60, t_sec) AS minute_sec, avg(map) AS map_avg, min(map) AS map_min, "
        "avg(hr) AS hr_avg, min(spo2) AS spo2_min FROM grid GROUP BY caseid, minute_sec WITH NO DATA",
    ]
    return tables, timescale


def unseen_ids():
    return sorted(int(c) for c in pd.read_parquet(PROCESSED / "preds_test.parquet", columns=["caseid"]).caseid.unique())


def frames(ids):
    """predictions, events and alarms of these test surgeries. p is null where not scored (preds_test already has it
    so). Alarms: evaluate.alarms on p10 with the val threshold, the rule the case files' alerts come from (only 50 test
    surgeries have a case file); is_true = an onset follows within 10 min, as evaluate counts it."""
    p = pd.read_parquet(PROCESSED / "preds_test.parquet", filters=[("caseid", "in", ids)],
                        columns=["caseid", "t_sec", f"eligible_{E}", "map_now", f"next_onset_{E}_s", f"p_snapshot_{E}_10",
                                 f"p_baseline1_{E}_10", *(f"p_fusion_{E}_{h}" for h in HS)]
                        ).sort_values(["caseid", "t_sec"], ignore_index=True)
    pred = pd.DataFrame({"caseid": p.caseid, "t_sec": p.t_sec, **{f"p{h}": p[f"p_fusion_{E}_{h}"] for h in HS},
                         "scored": p[f"eligible_{E}"], "map_now": p.map_now, "p_snapshot_10": p[f"p_snapshot_{E}_10"],
                         "p_baseline1_10": p[f"p_baseline1_{E}_10"]})
    thr = json.loads((PROCESSED / "models" / "threshold.json").read_text())[E]["threshold"]
    a = alarms(pred.p10.to_numpy(), pred.caseid.to_numpy(), pred.t_sec.to_numpy(), thr)
    al = pred.loc[a, ["caseid", "t_sec", "p10", "map_now"]].assign(is_true=p.loc[a, f"next_onset_{E}_s"] <= WIN)
    ev = pd.read_parquet(PROCESSED / "events.parquet", filters=[("caseid", "in", ids)],
                         columns=["caseid", "event", "onset_sec", "end_sec", "duration_s"])
    return {"predictions": pred, "events": ev, "alarms": al}


def csv(df):
    """COPY ... (FORMAT csv) text: NaN -> an empty unquoted field = NULL; floats in shortest round-trip form."""
    return df.to_csv(index=False, header=False)


def copy(conn, table, df):
    with conn.cursor() as cur, cur.copy(f"COPY {table} ({', '.join(df.columns)}) FROM STDIN (FORMAT csv)") as cp:
        cp.write(csv(df))


def load(schema=None, ids=None):
    """(Re)load the test surgeries: TRUNCATE, COPY in batches, refresh the 1-min rollup, then compress the chunks."""
    ids, dt = sorted(ids or unseen_ids()), grid_dtypes()
    tables, timescale = schema_sql(dt)
    with connect(schema) as conn:
        for s in tables:
            conn.execute(s)
        conn.execute("TRUNCATE grid, predictions, events, alarms")
        for s in timescale:
            conn.execute(s)
        n = 0
        for i in range(0, len(ids), BATCH):
            g = pd.concat([local_grid(c)[list(dt.index)].assign(caseid=c) for c in ids[i:i + BATCH]])
            copy(conn, "grid", g)
            n += len(g)
            print(f"grid: {min(i + BATCH, len(ids))}/{len(ids)} surgeries, {n:,} rows", flush=True)
        for name, df in frames(ids).items():
            copy(conn, name, df)
            print(f"{name}: {len(df):,} rows", flush=True)
        conn.execute("CALL refresh_continuous_aggregate('vitals_1min', NULL, NULL)")
        print("vitals_1min refreshed", flush=True)
        for h in ("grid", "predictions"):
            k = conn.execute(f"SELECT count(compress_chunk(c, if_not_compressed => TRUE)) FROM show_chunks('{h}') c")
            print(f"{h}: {k.fetchone()[0]} chunks compressed", flush=True)


QUERIES = [
    {"title": "Risk climbs before a drop",
     "plain": "The model's average 10-minute risk at each minute before a real hypotension onset (minute -15 to -1), "
              "against every other scored time with no drop in the next 15 minutes.",
     "sql": f"""WITH r AS (
  SELECT p.p10,
         (SELECT min(e.onset_sec) FROM events e
           WHERE e.caseid = p.caseid AND e.event = '{E}'
             AND e.onset_sec > p.t_sec AND e.onset_sec <= p.t_sec + {AHEAD}) - p.t_sec AS until_s
  FROM predictions p WHERE p.scored)
SELECT coalesce((time_bucket(60, -until_s) / 60)::text, 'no drop within 15 min') AS minute_before_onset,
       round(avg(p10)::numeric, 3) AS avg_risk_10min, count(*) AS prediction_times
FROM r GROUP BY 1 ORDER BY max(until_s) DESC NULLS LAST"""},
    {"title": "Alarms by the MAP when they fired",
     "plain": f"The model's alarms, grouped by the cleaned MAP the model read when each fired (no valid reading: not "
              f"recorded); 'under {LOW}' means already under {LOW}, but not yet for the 60 s an event needs. 'Followed by "
              f"a drop' means a real hypotension onset came within the next 10 minutes (the evaluation's true alarm).",
     "sql": f"""SELECT CASE WHEN map_now >= 75 THEN '75 or more' WHEN map_now >= {LOW} THEN '{LOW} to 75'
            WHEN map_now < {LOW} THEN 'under {LOW}' ELSE 'not recorded' END AS map_at_alarm,
       count(*) AS alarms, count(*) FILTER (WHERE is_true) AS followed_by_drop,
       round(100.0 * count(*) FILTER (WHERE is_true) / count(*), 1) AS percent_followed
FROM alarms GROUP BY 1 ORDER BY min(map_now) DESC NULLS LAST"""},
    {"title": f"Minutes under {LOW} per surgery",
     "plain": f"From the continuous aggregate vitals_1min (MAP rolled up per surgery-minute inside the database): the "
              f"10 surgeries with the most minutes of average MAP under {LOW}, next to their events and alarms.",
     "sql": f"""WITH m AS (
  SELECT caseid, count(*) FILTER (WHERE map_avg < {LOW}) AS minutes_under_{LOW}
  FROM vitals_1min GROUP BY caseid)
SELECT m.caseid, m.minutes_under_{LOW},
       (SELECT count(*) FROM events e WHERE e.caseid = m.caseid AND e.event = '{E}') AS hypotension_events,
       (SELECT count(*) FROM alarms a WHERE a.caseid = m.caseid) AS alarms,
       (SELECT count(*) FROM alarms a WHERE a.caseid = m.caseid AND a.is_true) AS alarms_followed_by_drop
FROM m ORDER BY m.minutes_under_{LOW} DESC, m.caseid LIMIT 10"""},
    {"title": "Compressed by surgery",
     "plain": "Every 2-second sample of every monitor signal the model reads, stored in hypertable chunks compressed "
              "column by column and segmented by surgery: size before and after.",
     "sql": """SELECT (SELECT count(*) FROM grid) AS grid_rows, s.total_chunks AS chunks,
       s.number_compressed_chunks AS compressed_chunks,
       round(s.before_compression_total_bytes / 1e6, 1) AS mb_before,
       round(s.after_compression_total_bytes / 1e6, 1) AS mb_after,
       round(s.before_compression_total_bytes::numeric / nullif(s.after_compression_total_bytes, 0), 1) AS ratio
FROM hypertable_compression_stats('grid') s"""},
]
_cache = {}


def run(conn, q):
    cur = conn.execute(q["sql"])
    return {**q, "columns": [d.name for d in cur.description],
            "rows": [[float(v) if isinstance(v, Decimal) else v for v in r] for r in cur]}


def run_all(schema=None):
    """Every insight query that succeeds (one that fails, e.g. on an older TimescaleDB, is left out with a warning)."""
    out = []
    with connect(schema) as conn:
        for q in QUERIES:
            try:
                out.append(run(conn, q))
            except psycopg.Error as e:
                warn(f"query '{q['title']}' failed", e)
    return out


def insights(schema=None):
    """GET /api/insights: cached for the process lifetime once the queries have run."""
    if not url():
        return {"available": False, "reason": f"{KEY} not set"}
    if schema not in _cache:
        try:
            out = run_all(schema)
        except psycopg.Error as e:
            warn("insights unavailable", e)
            return {"available": False, "reason": f"Tiger Cloud not reachable ({type(e).__name__})"}
        if not out:
            return {"available": False, "reason": "every Tiger query failed (has `python -m orew.tiger load` run?)"}
        _cache[schema] = {"available": True, "queries": out}
    return _cache[schema]


class GridSource:
    """The live engine's grid(caseid) from the Tiger hypertable: the same frame (columns, dtypes) as live.local_grid.
    Falls back to the local parquet without TIGER_DATABASE_URL, when the database fails, or for a surgery not loaded.
    name = "tiger" | "local" is what /api/health reports."""

    def __init__(self, schema=None):
        self.schema, self.name, self.dtypes = schema, "local", grid_dtypes()
        if url():
            try:
                with connect(schema) as conn:
                    if conn.execute("SELECT 1 FROM grid LIMIT 1").fetchone():
                        self.name = "tiger"
                    else:
                        print("tiger: grid is empty (run `python -m orew.tiger load`); local files used", file=sys.stderr)
            except psycopg.Error as e:
                warn("Tiger Cloud not used, local files instead", e)

    def __call__(self, caseid):
        if self.name == "tiger":
            try:
                with connect(self.schema) as conn:
                    rows = conn.execute(f"SELECT {', '.join(self.dtypes.index)} FROM grid WHERE caseid = %s "
                                        "ORDER BY t_sec", (caseid,)).fetchall()
                if rows:
                    return pd.DataFrame(rows, columns=self.dtypes.index).astype(self.dtypes.to_dict())
                print(f"tiger: surgery {caseid} not loaded; local file used", file=sys.stderr)
            except psycopg.Error as e:
                self.name = "local"  # stay local: mid-demo, a dead database would otherwise cost a timeout per surgery
                warn("grid query failed, local files from now on", e)
        return local_grid(caseid)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["load", "check"])
    a = ap.parse_args()
    try:
        if a.command == "load":
            load()
        with connect() as conn:
            for t in ("grid", "predictions", "events", "alarms", "vitals_1min"):
                print(f"{t}: {conn.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:,} rows")
        for q in run_all():
            print(f"\n{q['title']}\n  {' | '.join(q['columns'])}")
            for r in q["rows"]:
                print("  " + " | ".join(map(str, r)))
    except (psycopg.Error, RuntimeError) as e:
        sys.exit(f"tiger: {type(e).__name__}: {redact(str(e))}")


if __name__ == "__main__":
    main()
