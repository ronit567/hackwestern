from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RAW = ROOT / CFG["paths"]["raw"]
PROCESSED = ROOT / CFG["paths"]["processed"]
REPORTS = ROOT / CFG["paths"]["reports"]
