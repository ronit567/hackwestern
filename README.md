# Operating room early warning (VitalDB)

Every 30 s during surgery, the model estimates whether **hypotension** (arterial MAP under 65 mmHg for 60 s) or **hypoxemia** (SpO2 under 90 % for 2 min) starts in the next 5, 10 or 15 minutes, from the signals already on the anaesthesia monitor. An XGBoost snapshot model and a calibrated fusion stacker are compared against an alarm on the current number alone, including inside bands where that number still looks normal. The claim is **more time to act**, not better outcomes.

Binding spec: [SPEC.md](SPEC.md). Build plan and the results-page data contract: [PLAN.md](PLAN.md).

## Open the results page

```sh
uv run python -m orew.export                                    # reports/web/cohort.json + cases/*.json (metrics.json comes from evaluate)
uv run python -m http.server -d ~/Desktop/hackathon 8000        # then open http://localhost:8000/web/
```

- The page opens on the final test result, then a surgery explorer: vital signs over the whole surgery (h:mm), each hypotension event, and the model's first alarm in the 10 minutes before it ("warned 6.5 min earlier", the MAP at that alarm) or "missed". "Zoom in" on an event shows the 25 minutes around it.
- `#case=287` opens one surgery. The "Unseen test surgeries" cards (`web/featured.json`) pick four test surgeries, best and worst; `tests/test_export.py` checks each card's reason against the exported data.
- `?illustrative=1` replaces the hypoxemia blocks with **made-up** numbers (`web/illustrative_hypoxemia.json`) to show the layout with more events. Every such block, chart and table is labelled "ILLUSTRATIVE — NOT A REAL RESULT". Hypotension and the surgeries stay real.
- `?theme=light` or `?theme=dark` overrides the system colour scheme (projectors).

## Live demo

```sh
uv run python -m orew.app --port 8000      # then open http://localhost:8000/
```

- One server (FastAPI) for the whole demo: it serves the results page as before (`/web/`) and replays an unseen **test** surgery with the saved models running live. Every 30 s of surgery time, it computes the features from the monitor data up to that moment only, then the risk of hypotension in the next 5, 10 and 15 minutes, and fires the same alarm rule the evaluation used. You can watch the risk rise and the alarm fire minutes before MAP actually falls under 65, or see a drop the model missed (`1046`).
- The replayed numbers are not new results. For the surgeries tested, they equal the locked test predictions exactly (`tests/test_live.py`).
- API and event stream: [PLAN.md, "Live demo"](PLAN.md#live-demo).

## Final test results

Run once, locked (`reports/test.lock`): 140 surgeries from 139 patients never used for training, calibration or alert thresholds; 443 hypotension events; 53,269 prediction times. Hypotension, 10 minutes ahead. 95 % CIs from 1,000 patient-bootstrap resamples. Full tables: `reports/test_report.md` (written by `orew.evaluate`, not in git).

| Hypotension, 10 min | Fusion (ours) | Baseline 1: current MAP |
|---|---|---|
| AUROC | 0.842 (0.815–0.870) | 0.779 (0.750–0.809) |
| AUPRC | 0.373 (0.314–0.433) | 0.301 (0.250–0.357) |
| Brier score | 0.071 (0.059–0.084) | 0.076 (0.063–0.091) |
| Sensitivity at 90 % specificity | 0.545 (0.433–0.611) | 0.432 (0.384–0.522) |

Inside bands of the current MAP (AUROC). "Clear win": the paired-bootstrap 95 % interval of the gain is above zero.

| Current MAP, mmHg | Times (before an onset) | Fusion | Baseline 1 | Clear win |
|---|---|---|---|---|
| 65–70 | 4,754 (1,562) | 0.660 (0.607–0.711) | 0.605 (0.578–0.629) | **no** |
| 70–75 | 6,930 (1,007) | 0.686 (0.637–0.737) | 0.587 (0.564–0.613) | yes |
| 75–80 | 7,576 (624) | 0.712 (0.661–0.760) | 0.540 (0.511–0.570) | yes |
| > 80 | 28,492 (976) | 0.811 (0.758–0.862) | 0.578 (0.532–0.623) | yes |

Alerts: p10 crosses a threshold chosen on validation for at most 2 alarms per hour, then 5 min quiet unless risk rises by 0.10.

| Alerts | Fusion | Baseline 1 |
|---|---|---|
| Alarms per hour | 1.69 (1.33–2.06) | 1.60 (1.29–1.93) |
| False alarms per true alarm | 1.04 (0.81–1.35) | 1.23 (1.00–1.59) |
| Median lead of warned events, min | 4.50 (3.53–5.33) | 3.53 (2.63–4.43) |
| Events warned ≥ 5 min ahead | 31 % (23–38) | 29 % (21–35) |
| Event recall (alarm in the 10 min before) | 68 % (59–75) | 75 % (69–81) |

Hypoxemia had only 8 test events (64 in the whole cohort), so its numbers are exploratory and no claim is made.

## Limitations

- Retrospective, single centre (VitalDB, Seoul National University Hospital). External validation is needed.
- Arterial-line cohort: a fixed-seed sample of 1,000 adult general-anaesthesia cases with an arterial line (952 downloaded), sicker than average.
- Trade-off: the current-MAP alarm catches more drops (75 % against 68 %) but later; the model raises about one false alarm per true alarm.
- MAP 65–70 mmHg: higher AUROC, but not a clear win.
- Hypoxemia: too few events. The `?illustrative=1` numbers are made up and labelled as such.
- Not built yet: the trajectory specialist (GRU on trends + 1D CNN on the arterial wave) and the pattern model (autoencoder + k-means). They are the next steps; the results above are snapshot XGBoost + fusion.
- Vasopressor infusions are nearly absent from VitalDB and boluses carry no timestamps, so the drug ablation mostly measures propofol and remifentanil.
- Alarm reasons are SHAP drivers of the snapshot model in plain-language templates; a driver quoting an implausible level (an artifact, such as diastolic −30 mmHg) is skipped.

## Reproduce

```sh
uv sync
uv run python -m orew.download                # cohort + per-case cache in data/raw (~20 MB per .vital, deleted after extraction)
uv run python -m orew.labels                  # cleaned 2 s grids, events, prediction table, subject splits
uv run python -m orew.features                # snapshot features at every prediction time
uv run python -m orew.models                  # baselines, snapshot XGBoost, fusion + isotonic, val alert threshold
uv run python -m orew.evaluate --split val    # reports/val_report.md + reports/web/metrics.json
uv run pytest -q                              # leakage checklist and unit tests
uv run python -m orew.models --test           # once: test predictions (needs green tests; writes reports/TEST_LOCK)
uv run python -m orew.evaluate --split test   # once: reports/test_report.md (writes reports/test.lock)
uv run python -m orew.explain --split val     # alarm reasons; repeat with --split test
uv run python -m orew.export                  # page data
```

## Layout

```
config.yaml          every number: cohort, artifact rules, windows, alert rate, bins, bootstrap
src/orew/            download, clean, labels, features, models, evaluate, explain, export, illustrative; live + app (demo server)
tests/               leakage checklist (SPEC) and unit tests
web/                 static results page (index.html, app.js, style.css, featured.json); no build, no CDN
data/, reports/      local outputs, not in git
```
