# Plan: Operating Room Early Warning System

## Context
Build the model specified in *Operating Room Early Warning System: Project Brief* (PDF): every 30 s during surgery, predict whether **hypotension** (invasive MAP < 65 for ≥ 60 s) or **hypoxemia** (SpO2 < 90 for ≥ 2 min) starts in the next 5 / 10 / 15 min, on VitalDB. The design reuses the RescueWindow pattern of specialists plus a fusion layer. The claim to prove is: *it warns earlier and more accurately than watching the current number, including within bins where the number still looks normal.* The spec sections (Prediction task → Evaluation, Leakage checklist) are binding. The brief's own build plan is replaced by the plan below.

**User decisions:** multi-day scope (everything except MOVER, which needs a data use agreement); **~1,000 cases**; **no replay demo**.

**Facts checked live against the VitalDB API (2026-11-21):**
- Adult general-anesthesia cases with ≥ 30 min of recording: 5,988.
- Of those, 3,575 have `SNUADC/ART` + `Solar8000/ART_MBP` (the arterial cohort), and 5,986 have SpO2.
- `cases.csv` starts with a BOM, so read it with `utf-8-sig`. There are 6,090 unique `subjectid`s.
- A `.vital` file is about 21 MB per case (`https://api.vitaldb.net/{caseid}.vital`).
- `Orchestra/PHEN_RATE` exists in only 127 cases and `NEPI_RATE` in only 88, so vasopressor infusions are nearly absent.

**Environment:** empty `~/Desktop/hackathon`, Apple M4, 16 GB RAM, 239 GB free disk, `uv` installed, system Python 3.9 (uv will provide 3.11).

## Cohort decision
- Draw a fixed-seed random sample of **1,000 eligible arterial-cohort cases**. Use them for **both** events: every one has SpO2 and pleth, so the trajectory model's wave input is never missing.
- `n_cases` in the config can scale this up later.
- Caveat for the report: arterial-line patients are sicker than average, so the hypoxemia results describe that population.

## Layout (lean; one module per pipeline stage)
```
hackathon/                      # git init here (the current repo is wrongly rooted at ~)
  pyproject.toml                # uv, py3.11: vitaldb pandas numpy pyarrow scipy scikit-learn xgboost torch shap pyyaml matplotlib pytest
  config.yaml                   # horizons, windows, thresholds, artifact rules, seeds, n_cases, alarm rate
  src/orew/
    __init__.py                 # CFG = yaml load; paths
    download.py                 # cohort selection, parallel resumable download → per-case cache, delete .vital
    clean.py                    # 2 s grid, causal artifact rules, ffill ≤ 30 s, masks; per-beat ART + pleth features
    labels.py                   # event runs/onsets, prediction times, 5/10/15 labels, exclusions, flags, subject splits
    features.py                 # snapshot tabular features at t (+ drug-feature list for the ablation)
    crossfit.py                 # one generic 5-fold GroupKFold OOF helper used by every specialist
    baselines.py                # LR on current value; LR on value + 5-min slope
    snapshot.py                 # XGBoost per event × horizon + SHAP
    trajectory.py               # GRU (300×C trends + masks) ⊕ 1D CNN (60 s ART @ 100 Hz), 6 heads, MPS
    pattern.py                  # autoencoder embedding → K-Means (v2)
    fusion.py                   # stacker on OOF outputs + phase + current value; isotonic on val; alert threshold on val
    evaluate.py                 # metrics, alarm simulation, value bins, subgroups, bootstrap CIs, flagged on/off, report.md
    explain.py                  # SHAP → plain-language drivers; writes per-(case, t) JSON records
    export.py                   # reports/web/*.json for the results page (see "Results frontend")
  web/                          # static results page: index.html, app.js, style.css
  tests/test_leakage.py         # all 13 checklist items + truncation tests
  data/{raw,processed}/  reports/   # gitignored data
```
Each stage runs as `uv run python -m orew.<stage>` and reads and writes Parquet or `.npy` under `data/`. Nothing re-downloads inside training.

## Phases (stop and inspect at each ✋)

**0. Setup**
- `git init`, `uv init` with Python 3.11, add the dependencies, `.gitignore` for `data/` and `reports/`.
- Write `config.yaml` with every number from the brief.

**1. Data pull** (`download.py`)
- Select the cohort from the `/cases` and `/trks` CSVs: adult, general anesthesia, ≥ 30 min, arterial cohort. Sample 1,000 cases with a fixed seed.
- Load each case with `vitaldb.VitalFile`, then save:
  - numerics on a 2 s grid → `raw/{id}.parquet`;
  - the ART wave at native 500 Hz → per-beat features computed with `scipy.signal.find_peaks` (pulse pressure, max dP/dt, systolic area, beat-to-beat variability, breath-to-breath PP swing). **Each beat is stamped at its end time**, so it stays causal. Beats are binned to the 2 s grid;
  - the ART wave decimated to 100 Hz → `raw/{id}_art100.npy` (float32);
  - pleth amplitude per 2 s.
- Discard the `.vital` file after extraction. Pick up from the remaining cases if interrupted (skip cached ones), use 8 threads, retry failures.
- ✋ Do 5 cases first:
  - check that every track name in the brief exists;
  - **check how vitaldb aggregates when resampling numerics** (last value or mean), since that changes the 2 s MAP label;
  - plot MAP / HR / SpO2 to PNG and look at them;
  - then run the full 1,000.

**2. Cleaning + labels** (`clean.py`, `labels.py`)
- Apply the artifact rules **causally**: compare each point with the *previous* sample, never the next, so the same mask is valid for features and labels.
  - MAP outside 20–200.
  - Pulse pressure < 10.
  - |ΔMAP| > 40 within 2 s.
  - SpO2 < 50, or a sudden drop while pleth amplitude is low.
- Forward-fill gaps up to 30 s; longer gaps stay missing, with mask channels.
- Event runs:
  - hypotension: **invasive `ART_MBP` only**, < 65 for ≥ 30 consecutive 2 s steps;
  - hypoxemia: SpO2 < 90 for ≥ 2 min;
  - a run that is more than half artifact is not an event;
  - onset is the first second of the run.
- Prediction times:
  - every 30 s from recording start + 10 min;
  - drop t if an event is in progress, or if more than half the input window is missing or artifact;
  - flag t if it is within 5 min after an event ended;
  - label at each horizon H is 1 if an onset falls in (t, t+H].
- Split by `subjectid`: 70/15/15, fixed seed → `processed/splits.json`.
- ✋ Print event counts per cohort. Render 20 random events to PNG and inspect them for artifacts; tune the rules if needed.

**3. Leakage tests** (`tests/test_leakage.py`, written alongside phases 2–7, not after)
- Synthetic-signal unit tests cover the function-level items:
  - truncation (features and trajectory tensors at t are unchanged when data after t is deleted);
  - label and input windows don't overlap;
  - in-progress rows are dropped;
  - mostly-artifact runs never become events;
  - NIBP changes never change labels;
  - phase features are unchanged when future markers (`opend`/`aneend` > t) are set to NaN;
  - no `intraop_*` columns;
  - negatives are exactly the full 30 s grid minus documented exclusions.
- Tests on saved outputs cover the rest, and skip if those outputs aren't built yet:
  - no shared subjects between splits;
  - normalisation is fitted on train subjects only;
  - each OOF row was predicted by a fold that excluded its subject;
  - the threshold metadata says the threshold came from `val`;
  - the report contains value-bin tables with baseline 1.

**4. Snapshot features** (`features.py`)
- The groups in the brief's two feature tables:
  - pressure trend: mean / min / slope over 1, 3, 5, 10 min;
  - shock index;
  - beat features;
  - drugs and their 5-minute deltas;
  - BIS;
  - SpO2 trend, ETCO2 / RR / TV / PEEP / PIP / FiO2, pleth quality;
  - phase and minutes since phase start, using only markers already passed at t;
  - baseline MAP from the first 10 min after `anestart` (NaN before then);
  - safe preop clinical fields.
- Everything is computed from `grid[:t]` with trailing windows only.

**5. Evaluation + baselines** (`evaluate.py`, `baselines.py`)
- Metrics:
  - AUROC, AUPRC, Brier score and reliability plot;
  - sensitivity at 90% specificity;
  - simulated alarms per hour and false alarms per true alarm. An alarm fires when p10 crosses the threshold, then stays quiet for 5 min unless p10 rises by more than `rearm_delta`;
  - median lead time and share of events warned ≥ 5 min ahead;
  - event-level recall.
- Results inside current-value bins (MAP 65–70 / 70–75 / 75–80 / > 80; SpO2 90–94 / ≥ 95), with and without flagged times, with 1,000 patient-bootstrap 95% CIs, and by subgroup (age band, sex, ASA, department, emergency).
- Train both baselines and produce the full report on **val**. This sets the bar.

**6. Snapshot specialist** (`snapshot.py`, `crossfit.py`)
- XGBoost per event × horizon, with 5-fold grouped OOF on train.
- Training negatives are downsampled at random with sample weights. Evaluation always uses every eligible t.
- Run two feature sets: all features, and all minus drugs (for the ablation). Add SHAP.
- ✋ Must beat baseline 1 **within every bin** on val. If it doesn't, stop and look before going further.

**7. Trajectory specialist** (`trajectory.py`)
- A PyTorch Dataset slices memmapped per-case arrays on the fly, so the 16 GB of RAM is never filled.
- GRU over 300 steps × (channels + masks), concatenated with a 1D CNN over 6,000 wave samples, feeding 6 sigmoid heads.
- Normalisation statistics are fitted on train-fold subjects only. 5-fold OOF, run on MPS with a CPU fallback.

**8. Pattern model v2** (`pattern.py`)
- Autoencoder over the same trend windows. K-Means on embeddings of pre-event windows, with k = 3–6 chosen by silhouette score, fitted per fold.
- Cluster distances go to the stacker. Clusters are named only after inspecting example cases.

**9. Fusion** (`fusion.py`)
- Logistic regression (or a small XGBoost if it is clearly better on val) on the OOF specialist outputs, phase and current value, one per event × horizon.
- Isotonic calibration on val. Choose the alarm threshold on val at the alarm rate set in the config.
- `lead_specialist` is the specialist with the largest contribution to the stacker for that row.
- Full val report for every model, plus the drug ablation.

**10. Final test (once)** (`evaluate.py --split test`)
- Refuses to run unless `pytest tests/` passes, and writes a lock file so it can't run twice.
- Writes `reports/test_report.md`:
  - value-bin results against the baselines;
  - subgroups;
  - drug ablation;
  - alarms per hour;
  - lead time;
  - limits copied from the brief's Framing section.

**11. Output records** (`explain.py`)
- Writes the per-(case, t) JSON in the brief's exact schema: p5/p10/p15, `lead_specialist`, and 3 `top_drivers`.
- Drivers come from SHAP on the snapshot and fusion models, mapped through a dict of plain-language templates.

## Reuse instead of building
- `vitaldb` for loading and the track and case CSVs.
- `scipy.signal.find_peaks` / `decimate` for beat detection and the 100 Hz wave.
- sklearn for `GroupKFold`, `IsotonicRegression`, `LogisticRegression` and the metrics.
- `shap.TreeExplainer`.
- No custom frameworks. One `crossfit` helper serves all three specialists.

## Verification
- `uv run pytest tests/ -q`: all leakage tests pass before any number is reported (the final test run enforces this).
- ✋ checkpoints: I read the PNGs (5 case plots, 20 event plots) myself and report what I see.
- After each model, I report the val tables next to the baselines, overall and per bin, with CIs.
- Test split: touched once, at the end.

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Download is slow or flaky (~20–40 GB for 1,000 cases) | Med | Pick up where it stopped, 8 threads, retries; delete each `.vital` after extraction |
| vitaldb resampling semantics change the 2 s MAP label | Med | Verified on 5 cases in phase 1 before the full pull |
| Too few hypoxemia events in 1,000 arterial cases for useful CIs | Med–High | Count them in phase 2. If < ~150, add SpO2-only cases (numerics + pleth; the CNN input gets masked) |
| Vasopressor infusions are nearly absent; phenylephrine/ephedrine boluses have no timestamps | Certain | The drug ablation mostly measures propofol/remifentanil; say so in the report |
| Snapshot fails to beat baseline 1 in the MAP > 80 bin (few events) | Med | Report it honestly with CIs. Never tune on test |
| Causal artifact rules miss some artifact that future-aware rules would catch | Low–Med | Inspect 20 events. Labels may use stricter rules only if the truncation tests still pass |
| GRU on MPS gives slow or buggy results | Low | Fall back to CPU; keep the model small |

**Complexity: High** (multi-day). Skipped by decision: MOVER external validation and the replay demo.

## Results frontend
A static page (`web/index.html`, `app.js`, `style.css`: vanilla JS, hand-written SVG, no build, no CDN, works offline) that shows the cohort, the headline claim and single surgeries. There is no replay. It renders **only** what is in `reports/web/*.json`. If a file is missing, its section says which command to run.

```
uv run python -m orew.export                                   # cohort.json + cases/ (--max-cases, default 50)
uv run python -m http.server -d ~/Desktop/hackathon 8000       # then open http://localhost:8000/web/
```
`?theme=light` or `?theme=dark` overrides the system colour scheme (projectors, screenshots). The page computes nothing itself except one count: in how many value bands the final model's AUROC point estimate beats baseline 1's. `uv run pytest tests/test_export.py` checks the JSON and that `web/app.js` parses (`node --check`).

**Conventions:** JSON written with `allow_nan=False` (missing = `null`). Times are **minutes from `casestart`**. Probabilities and rates are fractions 0–1 (the page shows %). Every metric is `[estimate, ci_low, ci_high]` (95 % CI, patient bootstrap) or `null` if not computed. Object keys for horizons are strings: `"5"`, `"10"`, `"15"`. Events are `"hypotension"` and `"hypoxemia"`. Models are `baseline1, baseline2, snapshot, trajectory, pattern, fusion`.

### `reports/web/cohort.json`: `export.py`
| Field | Shape / unit | Source |
|---|---|---|
| `n_cohort`, `n_cached`, `n_subjects` | int | `cohort.csv` and cached `raw/{id}.parquet` |
| `age` | `{min, q1, median, q3, max}` years | `cases.csv`, cached cases |
| `sex`, `asa`, `department` | `{value: count}`; missing → `"unknown"` | ″ |
| `emergency` | `{n, of}` (`emop == 1`) | ″ |
| `median_duration_min` | float, `caseend − casestart` | ″ |
| `signals` | `[{name, pct_time, pct_cases}]`: % of 2 s grid rows with data (mean over cases) and % of cases with any data, for each parquet column | cached parquets |
| `n_labelled`, `events` *(once labels.py has run)* | int (cached cases with a `processed/grid` file); `{event: {n_events, n_cases}}` over those cases | `processed/events.parquet` |
| `splits` *(once labels.py has run)* | `{train|val|test: {n_subjects, n_cases}}` over the whole cohort | `processed/splits.json` |

### `reports/web/cases/index.json` + `cases/{caseid}.json`: `export.py`
- `index.json`: `[{caseid, duration_min, age, sex, height, weight, bmi, asa, emop, department, optype, opname}]` for the first `--max-cases` cached cases, in cohort order, plus `split` (`null` until labels.py has run).
- `{caseid}.json`:

| Field | Shape / unit |
|---|---|
| `caseid`, `duration_min` | int, float min |
| `demographics` | the same clinical fields as an index entry |
| `phases_min` | `{anestart, opstart, opend, aneend}` min (null if missing) |
| `t_min` | `[n]` min. 10 s bins `(t − 10 s, t]`, each holding the last finite 2 s value (raw grid, before artifact cleaning) |
| `series` | `{map, nibp_map, spo2, hr, etco2, bis, ppf_rate, rftn_ug_h}`, each `[n]` aligned with `t_min`, `null` where the bin has no data. Units: mmHg, mmHg, %, bpm, mmHg, BIS 0–100, mL/h (propofol 20 mg/mL), µg/h (remifentanil, both pumps: 20 × `rftn_rate` + 50 × `rftn50_rate`, as in `clean.py`) |
| `split` *(labels.py)* | `"train" | "val" | "test"`, from `processed/splits.json`. The page says what the split means for the risk traces (see below) |
| `events` *(labels.py)* | `[{type, onset_min, end_min}]` from `processed/events.parquet` (`end_min` = last grid time of the run). Present only for labelled cases (`processed/grid/{id}.parquet` exists); `[]` means labelled with no events |
| `risk` *(later, fusion/explain.py)* | `{event: {t_min: [m], p5: [m], p10: [m], p15: [m], threshold: float}}` on the 30 s prediction grid, `null` at dropped times |
| `alerts` *(later, explain.py)* | `[{type, t_min, p10, lead_specialist, top_drivers: [str, str, str]}]`: fired alerts (after the 5 min refractory period) with the plain-language drivers from that record |

`export.py` adds `split` and `events` whenever the labels.py outputs exist. `risk` and `alerts` are for explain.py's records: when they exist, `export.py` must add them in this shape, with these split rules:
- **train**: only from a cross-fitted stacker, so every train surgery is scored by specialist *and* stacker folds that excluded its subject. If fusion.py fits one stacker on all of train, train surgeries get no `risk` or `alerts`.
- **val**: allowed. The page notes that isotonic calibration and the alert threshold were fit on this split.
- **test**: dropped by `export.py` until `evaluate.py --split test` has written its lock file, so no test-patient predictions are viewed before the locked run.

The page already draws all four: shaded events, risk panels with the threshold, alert triangles and the driver list. Without them it says what is missing.

### `reports/web/metrics.json`: written by `evaluate.py` (`--split val|test`)
```
{
  "split": "val",                      # "val" | "test"
  "final_model": "fusion",             # compared with baseline1 in the headline; any computed model
  "models": [...],                     # models computed, in display order
  "horizons_min": [5, 10, 15], "headline_horizon_min": 10, "bootstrap": 1000,
  "n": {event: {cases, subjects, times, events}},
  "alerts": {"chosen_on": "val", "alarms_per_hour_target": 2.0,
             "threshold": {event: {model: p10_threshold}}},        # every model alarms at the same rate
  "overall":              {event: {H: {model: {metric: [e, lo, hi]}}}},
  "overall_excl_flagged": same shape, times flagged within 5 min after an event dropped,
  "bins":        {event: {H: [{label: "65-70", lo: 65, hi: 70, n, n_pos,
                                models: {model: {auroc: [e, lo, hi], auprc: [e, lo, hi]}}}]}},
  "subgroups":   {event: {H: {attr: [{label, n, n_pos, models: {model: {auroc, auprc}}}]}}},
  "reliability": {event: {H: {model: [{p, obs, n}]}}},            # mean predicted, observed rate, count per bin
  "ablation":    {event: {H: {model: "snapshot", with_drugs: {auroc, auprc}, without_drugs: {auroc, auprc}}}}
}
```
- `metric` ∈ `auroc, auprc, brier, sens_at_90spec, alarms_per_hour, false_per_true_alarm, median_lead_min, pct_warned_5min, event_recall`. The alert metrics come from the p10 alert simulation (threshold from `alerts`, 5 min refractory period, `rearm_delta`). The page reads them at the headline horizon.
- `bins`: MAP `65-70, 70-75, 75-80, >80` (hypotension) and SpO2 `90-94, >=95` (hypoxemia), from `config.yaml` `eval.*_bins` (`lo` inclusive, `hi` exclusive), binned on the current value at t.
- `subgroups` attrs: `age_band` (`eval.age_bands`), `sex`, `asa`, `department`, `emergency`.
- `split: "test"` is written only by the single locked test run. The page shows the split in the headline.
