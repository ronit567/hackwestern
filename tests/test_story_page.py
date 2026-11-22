"""The story page (web/demo.*): parses, loads only local files, has no replay, and types in no result."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
HTML, JS, CSS = (WEB / f"demo.{e}" for e in ("html", "js", "css"))
METRICS = ROOT / "reports/web/metrics.json"


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_parses():
    subprocess.run(["node", "--check", str(JS)], check=True)


def test_only_local_files():
    html = HTML.read_text()
    for ref in re.findall(r'(?:src|href)="([^"]*)"', html):
        assert not re.match(r"[a-z]+:|//", ref), ref
        assert ref.startswith("#") or (WEB / ref).exists() or ref == "live.html", ref  # live.html: the monitor agent's page
    assert "@import" not in CSS.read_text() and "url(" not in CSS.read_text()
    assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", JS.read_text())


def test_sections_and_links():
    html = HTML.read_text()
    for sid in ("top", "how", "honest", "results", "tiger", "limits"):
        assert f'id="{sid}"' in html
    assert html.count('href="live.html"') >= 2 and 'href="index.html"' in html
    js = JS.read_text()
    assert "EventSource" not in js and "replay/" not in js  # the replay lives in live.html


def test_no_result_typed_in():
    m = json.loads(METRICS.read_text())
    H = str(m["headline_horizon_min"])
    text = HTML.read_text() + JS.read_text()
    for model in (m["final_model"], "baseline1"):
        for k, v in m["overall"]["hypotension"][H][model].items():
            if v is None:
                continue
            e = v[0]
            forms = {f"{e:.3f}", f"{e:.2f}"} if e < 1 else {f"{e:.1f} min"}
            if k in ("event_recall", "sens_at_90spec", "pct_warned_5min"):
                forms |= {f"{e * 100:.0f} %", f"{e * 100:.0f}%"}
            for f in forms:
                assert f not in text, (model, k, f)


def test_constants_match_their_sources():
    js = JS.read_text()
    snap = json.loads((ROOT / "data/processed/models/snapshot_hypotension_10.json").read_text())
    assert f"N_FEATURES = {len(snap['learner']['feature_names'])};" in js
    spec = (ROOT / "SPEC.md").read_text().split("## Leakage checklist")[1].split("\n## ")[0]
    n_leak = len(re.findall(r"^\d+\. ", spec, re.M))
    assert f"N_LEAK = {n_leak};" in js
