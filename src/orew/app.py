"""Live demo server: the results pages and the live replay API (PLAN.md "Live demo").

uv run python -m orew.app --port 8000      # then open http://localhost:8000/
"""
import argparse
import asyncio
import json
import math
import os
import re
import threading

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from orew import REPORTS, ROOT
from orew.evaluate import STEP as TICK
from orew.live import VSTEP, LiveModel

HISTORY_S = 900  # a stream opened at start_sec > 0 first sends the 15 min of vitals before it


def load_env(path=ROOT / ".env"):
    """KEY=VALUE lines of the gitignored .env into os.environ; the real environment wins. Values are never printed."""
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, allow_nan=False)}\n\n"


def create_app(model=None, warm=False):
    """warm: compute the featured surgeries' replays in the background at start-up, so the demo opens instantly."""
    model = model or LiveModel()
    cache, lock = {}, threading.Lock()

    def replay(cid):
        with lock:  # one surgery computed at a time (the laptop runs hot); then cached for every later stream
            if cid not in cache:
                cache[cid] = model.replay(cid)
        return cache[cid]

    if warm:
        featured = [s["caseid"] for s in model.surgeries() if s["featured"]]
        threading.Thread(target=lambda: [replay(c) for c in featured], daemon=True).start()

    app = FastAPI(title="OR early warning: live demo")

    @app.get("/")
    def root():
        return RedirectResponse("/web/demo.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "models": True, "data_source": model.data_source}

    @app.get("/api/surgeries")
    def surgeries():
        return {"surgeries": model.surgeries()}

    @app.get("/api/insights")
    def insights():
        if not os.environ.get("TIGER_DATABASE_URL"):
            return {"available": False, "reason": "TIGER_DATABASE_URL not set"}
        try:
            from orew.tiger import insights as tiger_insights  # the Tiger Cloud integration's showcase queries
        except ImportError:
            return {"available": False, "reason": "Tiger queries not built yet (orew.tiger)"}
        return tiger_insights()

    @app.get("/api/replay/{caseid}")
    async def replay_stream(caseid: str, request: Request, speed: float = 60.0, start_sec: int = 0):
        # only an id from the test list ever reaches a file path
        if not re.fullmatch(r"[0-9]{1,6}", caseid) or int(caseid) not in model.pt:
            raise HTTPException(404, "not a test-split surgery")
        cid, speed = int(caseid), 60.0 if math.isnan(speed) else min(max(speed, 1.0), 600.0)

        async def events():
            yield ": computing\n\n"  # open the stream at once; a surgery's first replay takes a few seconds
            r = await asyncio.to_thread(replay, cid)
            start = min(max(start_sec, 0), r["meta"]["duration_s"])
            yield sse("meta", {**r["meta"], "speed": speed, "start_sec": start})
            prev = None
            for s in r["steps"]:
                if s["t_sec"] < start:
                    continue
                if await request.is_disconnected():
                    return
                if prev is None:
                    prev = s["t_sec"] - HISTORY_S if start > 0 else -1
                v = r["vitals"][max(prev // VSTEP + 1, 0): s["t_sec"] // VSTEP + 1]  # vitals in (prev, t]
                yield sse("tick", {"t_sec": s["t_sec"], "vitals": v, "p5": s["p5"], "p10": s["p10"], "p15": s["p15"],
                                   "scored": s["scored"], "map_now": s["map_now"]})
                if s["alarm"]:
                    yield sse("alarm", s["alarm"])
                for o in s["onsets"]:
                    yield sse("onset", o)
                prev = s["t_sec"]
                await asyncio.sleep(TICK / speed)
            yield sse("end", {"summary": r["summary"]})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # the layout `python -m http.server -d <repo>` gave the results page; nothing else of the repo is served
    app.mount("/web", StaticFiles(directory=ROOT / "web", html=True), name="web")
    app.mount("/reports/web", StaticFiles(directory=REPORTS / "web"), name="reports")
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    load_env()
    uvicorn.run(create_app(warm=True), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
