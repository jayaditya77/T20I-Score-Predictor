# T20I Score Predictor — Cricsheet Edition

An interview-ready machine-learning application that predicts a men's T20 International first-innings final score from the live match state.

## Dataset

This version explicitly uses **Cricsheet**. The setup script downloads the official T20 International JSON archive:

`https://cricsheet.org/downloads/t20s_json.zip`

The training pipeline filters to men's T20 internationals, excludes shortened/DLS-style matches, and uses first-innings ball-by-ball states. Cricsheet JSON is the official/main Cricsheet format.

## ML approach

- Target: **remaining runs**, not final score directly
- Final prediction: `current_score + predicted_remaining_runs`
- Model: **XGBoost Regressor**
- Validation: **match-level 80/20 holdout** to prevent leakage between states from the same match
- Separate **late-innings model** for the final five overs
- Rare city/venue labels are grouped into `Other`
- Very-late predictions use a small train-set state-median calibration for stability

### Features

- batting team
- bowling team
- city / venue region
- current score
- balls left
- wickets left
- current run rate
- runs in the recent 5-over window
- recent run rate
- innings phase
- baseline run rate
- baseline remaining runs
- recent-rate projected final score
- recent-vs-current run-rate gap

## Important input convention

The app asks for **legal balls completed in the current over**.

- `10 overs + 0 balls` → 60 legal balls
- `10 overs + 1 ball` → 61 legal balls
- ...
- `10 overs + 5 balls` → 65 legal balls
- There is **no 10.6 state**. After the sixth legal ball, enter `11 completed overs + 0 balls`.

The app validates this again when the prediction form is submitted, so an old/stale Streamlit widget value cannot silently reach the model.

## Why the app does not cap predictions at 6 × balls remaining

A legal-ball count is not a strict mathematical upper bound on T20 runs because wides, no-balls and overthrows can add runs without consuming a legal ball in the normal way. The app therefore keeps the model prediction non-negative without imposing a false hard six-runs-per-legal-ball ceiling.

## Validation and safety checks

Before prediction the app checks:

- legal balls in current over are 0–5
- completed overs are 2–20
- 20 overs requires 0 current-over balls
- wickets lost are 0–10
- last-five-overs runs cannot exceed the current score
- before five legal overs, the recent-window score must equal the innings score
- total legal balls cannot exceed 120

If 20 overs are complete or all 10 wickets are lost, the app returns the current score rather than asking the model to predict beyond the innings.

## Run on macOS / Linux

Python 3.11 is recommended.

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python setup_cricsheet.py
python -m streamlit run app.py
```

The current requirements pin the tested package versions for reproducibility.

`setup_cricsheet.py`:

1. downloads and extracts the official Cricsheet T20I JSON archive;
2. parses the raw deliveries and engineers match-state features;
3. creates a match-level 80/20 train/test split;
4. trains the overall and late-innings XGBoost models;
5. writes the actual held-out metrics and model metadata to `artifacts/`.

If the JSON archive is already extracted, the setup script does not download it again unless you delete `data/raw/t20s_json`.

## What to say in an interview

> I sourced men's T20 International ball-by-ball data from Cricsheet, parsed the raw JSON deliveries, engineered match-state and recent-form features, and trained XGBoost to predict remaining runs. I used a match-level holdout split to prevent leakage and a dedicated late-innings model for the final five overs. The Streamlit application validates the live match state, loads the trained artifacts, and displays held-out MAE, RMSE and R² from the training pipeline.

Do not quote a metric unless it is the metric produced by the current trained artifacts. For the Cricsheet snapshot used during development, the recorded held-out metrics were MAE 15.16 runs, RMSE 20.89 runs, R² 0.806, and late-innings MAE 6.87 runs.

## Source / attribution

Cricsheet: https://cricsheet.org/

Downloads: https://cricsheet.org/downloads/

Format documentation: https://cricsheet.org/format/

Please retain Cricsheet attribution when publishing or redistributing work produced from the dataset.
