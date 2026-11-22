"""The live monitor page (web/live.*): the script parses, no result number is typed in, only local files are loaded."""
import json
import re
import shutil
import subprocess

import pytest

from orew import PROCESSED, REPORTS, ROOT

WEB = ROOT / "web"
FILES = {n: (WEB / n).read_text() for n in ("live.html", "live.js", "live.css")}
# display constants the page may use: chart ticks 0.2..0.8, the "<1%" cut-off, the 1-point trend dead band, sparkline pad 0.1
ALLOWED = {"0.005", "0.01", "0.1", "0.2", "0.4", "0.6", "0.8"}


def numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from numbers(v)
    elif isinstance(obj, float):
        yield obj


def test_script_parses():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    subprocess.run([node, "--check", str(WEB / "live.js")], check=True)


def test_no_metric_hard_coded():
    js = FILES["live.js"] + re.sub(r"<svg.*?</svg>", "", FILES["live.html"], flags=re.S)  # icon paths are geometry
    stray = set(re.findall(r"(?<![\w.#-])\d*\.\d+(?![\w.])", js)) - ALLOWED
    assert not stray, f"decimal literals in the monitor: {stray}"
    thr = json.loads((PROCESSED / "models" / "threshold.json").read_text())["hypotension"]["threshold"]
    metrics = json.loads((REPORTS / "web" / "metrics.json").read_text())
    for v in [thr, *numbers(metrics)]:
        for d in (2, 3):
            s = f"{v:.{d}f}"
            if s not in ALLOWED and not s.endswith("0"):
                assert s not in js, f"{s} (a result number) is typed into the monitor"


def test_only_local_files():
    refs = re.findall(r"""(?:src|href)=["']([^"']+)""", FILES["live.html"])
    assert refs and all(not re.match(r"(?:[a-z]+:)?//", r) for r in refs), refs
    for name, text in FILES.items():
        urls = set(re.findall(r"https?://[^\s\"'`)]+", text)) - {"http://www.w3.org/2000/svg"}
        assert not urls, f"{name} loads {urls}"
        assert "@import" not in text and "fetch(\"http" not in text
    assert 'href="live.html"' in (WEB / "index.html").read_text()  # the results page links to the monitor
