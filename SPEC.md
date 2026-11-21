# SPEC — Operating Room Early Warning System

Condensed from the project brief (Nov 21, 2026). This file is binding; `PLAN.md` says how we build it. All numbers live in `config.yaml`.

## Task
At each prediction time t (every 30 s, starting once a full 10-min input window exists) the model sees only signals at or before t and outputs the probability that each event **starts** in (t, t+H], for H = 5, 10 (headline), 15 min.

| Event | Signal | Definition | Onset |
|---|---|---|---|
| Hypotension | invasive `Solar8000/ART_MBP` only (never NIBP) | 2 s MAP < 65 mmHg for ≥ 60 s | first second of the run |
| Hypoxemia | `Solar8000/PLETH_SPO2` | SpO2 < 90 % for ≥ 2 min | first second of the run |

**Sampling:**
- Use every eligible prediction time. Never pick negatives only from stable stretches.
- Training may downsample negatives *at random* with sample weights. Evaluation uses every eligible t.

**Exclusions and flags for each t:**
- Drop t if the event is in progress at t.
- Drop t if more than half of the input window is missing or artifact.
- Flag t if it falls within 5 min after an event ended. Report metrics with and without flagged times.

**Artifact rules** (starting point, tune by inspection):
- MAP outside 20–200 mmHg.
- Flat or square ART waveform: pulse pressure < 10 mmHg, or MAP jumping > 40 mmHg within 2 s.
- SpO2 < 50 %, or a sudden SpO2 drop while pleth is poor.
- A run that is mostly artifact never counts as an event.

## Data (VitalDB, open)
- Python: `vitaldb.find_cases`, `vitaldb.load_case(caseid, tracks, interval)`, `vitaldb.VitalFile`.
- Web API: `https://api.vitaldb.net/cases`, `/trks`. The `.vital` file for each case is `https://api.vitaldb.net/{caseid}.vital` (~20 MB). `cases.csv` has a BOM, so read it with `utf-8-sig`.
- Clinical times are seconds from case start.
- Cache every case locally (Parquet / npy). Never re-download inside training.

**Tracks:**

| Signal | Tracks |
|---|---|
| Arterial waveform | `SNUADC/ART` (500 Hz) |
| Arterial pressure | `Solar8000/ART_MBP`, `ART_SBP`, `ART_DBP` |
| Non-invasive MAP | `Solar8000/NIBP_MBP` (backup only, never for labels) |
| Heart rate | `Solar8000/HR` |
| ECG | `SNUADC/ECG_II` (not used) |
| SpO2 | `Solar8000/PLETH_SPO2` |
| Pleth waveform | `SNUADC/PLETH` |
| End-tidal CO2 | `Solar8000/ETCO2` or `Primus/ETCO2` |
| Respiratory rate | `Solar8000/RR_CO2` or `Primus/RR_CO2` |
| Ventilator | `Primus/FIO2`, `Primus/TV`, `Primus/PEEP_MBAR`, `Primus/PIP_MBAR` |
| Volatile agent | `Primus/EXP_SEVO`, `Primus/EXP_DES`, `Primus/MAC` |
| Infusions | `Orchestra/PPF20_RATE`, `Orchestra/RFTN20_RATE` or `RFTN50_RATE`, `Orchestra/PHEN_RATE`, `Orchestra/NEPI_RATE` |
| Depth and temperature | `BIS/BIS`, `Solar8000/BT` |

**Cohort** (our decision): adults (age ≥ 18), `ane_type == "General"`, ≥ 30 min recording, has `SNUADC/ART` and `Solar8000/ART_MBP`. Fixed-seed random sample of `n_cases` (1,000). The same cases are used for both events.

**Clinical fields:**
- Safe as features: `age, sex, height, weight, bmi, asa, emop, department, optype, ane_type, preop_*`.
- Phase markers `anestart, opstart, opend, aneend`. Induction is anestart→opstart, maintenance is opstart→opend, emergence is opend→aneend.
  - Use them only as "time since <marker already passed at t>".
  - Never "time until" a marker. Never reveal a marker that is still in the future at t.
- **Never** use any `intraop_*` field (whole-surgery totals, they leak the outcome).

## Preprocessing
- Resample numerics to a 2 s grid. Forward-fill gaps ≤ 30 s; longer gaps stay NaN, with a mask feature.
- Fit normalisation and imputation statistics on training patients only.
- Use the ART waveform at 100 Hz for the waveform model.

## Features (only data ≤ t)
**Hypotension:**
- MAP, SBP and DBP: current, mean, min, and slope over 1, 3, 5 and 10 min.
- HR level and slope; shock index (HR / SBP).
- Per-beat ART features: pulse pressure, max dP/dt, systolic area, beat-to-beat variability, breath-to-breath pulse-pressure swing.
- Drugs: propofol and remifentanil rates and their change over the last 5 min, vasopressor rates, volatile agent and MAC.
- BIS level and slope.
- Context: phase, minutes since `anestart`, age, ASA, BMI, emergency flag, baseline MAP from the first minutes.

**Hypoxemia:**
- SpO2: current, min, and slope over 1, 3 and 5 min.
- ETCO2 level and change, RR, TV, PEEP, PIP, FiO2.
- Pleth amplitude and stability.
- Phase and minutes since phase start.
- BMI, age, ASA, `preop_hb`, `preop_sao2`.

## Models
| Component | Input | Method |
|---|---|---|
| Baseline 1 | current MAP (or SpO2) | logistic regression |
| Baseline 2 | current value + 5-min slope | logistic regression |
| Snapshot specialist | tabular features at t | XGBoost, one per event × horizon; SHAP drivers |
| Trajectory specialist | 10 min of numerics on a 2 s grid (300 steps) + masks as channels, and 60 s of ART at 100 Hz | GRU on the trends ⊕ 1D CNN on the wave, shared heads for both events × 3 horizons |
| Pattern model (v2) | same trends | autoencoder embedding → K-Means; name clusters only after inspecting cases |
| Fusion stacker | out-of-fold specialist outputs + phase + current values | LR or small XGBoost per event × horizon; isotonic calibration on val |

## Evaluation
**Splits:**
- Split by `subjectid`, 70 / 15 / 15, fixed seed, saved to disk. Test is touched once.
- Base models: 5-fold grouped CV on train → out-of-fold predictions for the stacker.

**Baselines:** baseline 1, baseline 2 and each specialist alone, all reported next to the final model.

**Honest test:**
- Report results within bins of the current value: MAP 65–70 / 70–75 / 75–80 / > 80; SpO2 90–94 / ≥ 95.
- The model must beat baseline 1 **inside each bin**.

**Metrics:**
- AUROC and AUPRC.
- Reliability plot and Brier score.
- Sensitivity at 90 % specificity.
- Alarms per hour; false alarms per true alarm.
- Median lead time; share of events warned ≥ 5 min ahead.
- Event-level recall: share of events with an alert in the 10 min before onset, with repeat alarms within 5 min counted once.

**Reporting:**
- 95 % CIs from 1,000 bootstrap resamples of patients.
- Subgroups: age bands, sex, ASA, department, emergency vs planned.
- Drug-feature ablation: results with and without drug features.

## Output (one record per prediction time)
```json
{"caseid": 1234, "t_sec": 1830,
 "hypotension": {"p5": 0.41, "p10": 0.72, "p15": 0.80, "lead_specialist": "trajectory",
                 "top_drivers": ["map_slope_5m", "ppf_rate_delta_5m", "pulse_pressure"]},
 "hypoxemia":   {"p5": 0.03, "p10": 0.05, "p15": 0.07, "lead_specialist": "snapshot",
                 "top_drivers": ["spo2_slope_3m", "etco2", "phase_emergence"]}}
```
- Drivers come from SHAP on the snapshot model and the stacker, mapped to plain-language templates.
- An alert fires when p10 crosses a threshold chosen on **val** at a fixed alarm rate. It then stays quiet for 5 min unless risk rises further.

## Leakage checklist (each is an automated test in `tests/`; no result is reported until all pass)
1. No `subjectid` appears in more than one split.
2. Features at t are unchanged when all data after t is deleted (truncation test).
3. The label window (t, t+H] never overlaps the input window.
4. No `intraop_*` field is used as a feature.
5. Phase features only use markers already passed at t.
6. Negatives are every eligible prediction time.
7. Prediction times with an event in progress are dropped.
8. Normalisation and imputation are fit on training patients only.
9. The stacker trains only on out-of-fold predictions.
10. The alert threshold is chosen on val, never on test.
11. Runs that are mostly artifact are never labelled as events.
12. Hypotension labels use invasive MAP only, never NIBP.
13. Results are reported within current-value bins, against the current-value baseline.

## Framing
- The claim is "more time to act", not "saves lives".
- Results are retrospective and from a single centre; external validation is needed.
- Prior work exists (the Hypotension Prediction Index, Prescience). Our contribution is an open build with an evaluation that rules out the current-value shortcut.
