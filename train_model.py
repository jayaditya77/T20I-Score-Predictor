from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from utils import phase_name

BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "data" / "raw" / "t20s_json"
ARTIFACT_DIR = BASE / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
MIN_BALLS_BOWLED = 12
ROW_STRIDE = 2  # use every second legal ball to keep training fast on a laptop
MIN_CITY_MATCHES = 8

CAT_COLS = ["batting_team", "bowling_team", "city", "phase"]
NUM_COLS = [
    "current_score", "balls_left", "wickets_left", "crr", "last_five",
    "recent_rr", "baseline_rr", "baseline_remaining",
    "recent_projected_final", "recent_gap",
]
FEATURES = CAT_COLS + NUM_COLS

NON_WICKET_KINDS = {"retired hurt"}


def legal_delivery(delivery: dict) -> bool:
    extras = delivery.get("extras", {}) or {}
    return "wides" not in extras and "noballs" not in extras


def wicket_count(delivery: dict) -> int:
    total = 0
    for w in delivery.get("wickets", []) or []:
        if str(w.get("kind", "")).lower() not in NON_WICKET_KINDS:
            total += 1
    return total


def match_is_usable(match: dict) -> bool:
    info = match.get("info", {})
    if str(info.get("match_type", "")).upper() != "T20":
        return False
    if str(info.get("gender", "")).lower() != "male":
        return False
    # D/L, DLS and similar shortened matches make a fixed 120-ball target inconsistent.
    outcome = info.get("outcome", {}) or {}
    if outcome.get("method"):
        return False
    innings = match.get("innings", []) or []
    return len(innings) >= 1


def first_innings_states(match_id: str, match: dict) -> tuple[list[dict], dict] | tuple[None, None]:
    if not match_is_usable(match):
        return None, None

    info = match["info"]
    innings = match["innings"][0]
    batting_team = innings.get("team")
    teams = info.get("teams", []) or []
    if not batting_team or len(teams) != 2:
        return None, None
    bowling_team = teams[1] if teams[0] == batting_team else teams[0]
    city = info.get("city") or info.get("venue") or "Other"

    # First pass: get the final score and legal-ball total.
    final_score = 0
    legal_balls_total = 0
    final_wickets = 0
    for over in innings.get("overs", []) or []:
        for delivery in over.get("deliveries", []) or []:
            final_score += int((delivery.get("runs", {}) or {}).get("total", 0))
            final_wickets += wicket_count(delivery)
            if legal_delivery(delivery):
                legal_balls_total += 1

    # Exclude obviously shortened/abandoned first innings unless all out.
    if legal_balls_total < 90 and final_wickets < 10:
        return None, None
    if legal_balls_total > 120:
        # Regulation T20 first innings should have at most 120 legal balls.
        legal_balls_total = 120

    rows = []
    score = 0
    wickets = 0
    legal_balls = 0
    score_after_legal_ball = [0]

    for over in innings.get("overs", []) or []:
        for delivery in over.get("deliveries", []) or []:
            score += int((delivery.get("runs", {}) or {}).get("total", 0))
            wickets += wicket_count(delivery)
            if not legal_delivery(delivery):
                continue

            legal_balls += 1
            score_after_legal_ball.append(score)
            if legal_balls < MIN_BALLS_BOWLED or legal_balls > 120:
                continue
            if legal_balls % ROW_STRIDE != 0 and legal_balls != 120:
                continue

            balls_left = 120 - legal_balls
            wickets_left = max(0, 10 - wickets)
            crr = score * 6.0 / legal_balls
            lookback = min(30, legal_balls)
            score_lookback = score_after_legal_ball[legal_balls - lookback]
            last_five = score - score_lookback
            recent_rr = last_five * 6.0 / lookback
            baseline_rr = 0.65 * recent_rr + 0.35 * crr
            baseline_remaining = baseline_rr * balls_left / 6.0
            recent_projected_final = score + recent_rr * balls_left / 6.0
            recent_gap = recent_rr - crr

            rows.append({
                "match_id": match_id,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "city": city,
                "current_score": score,
                "balls_left": balls_left,
                "wickets_left": wickets_left,
                "crr": crr,
                "last_five": last_five,
                "recent_rr": recent_rr,
                "phase": phase_name(legal_balls),
                "baseline_rr": baseline_rr,
                "baseline_remaining": baseline_remaining,
                "recent_projected_final": recent_projected_final,
                "recent_gap": recent_gap,
                "remaining_runs": max(0, final_score - score),
            })

    match_meta = {
        "match_id": match_id,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city": city,
        "final_score": final_score,
    }
    return rows, match_meta


def load_dataset() -> pd.DataFrame:
    json_files = sorted(RAW_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No Cricsheet JSON files found in {RAW_DIR}. Run: python download_cricsheet.py"
        )

    all_rows = []
    match_metas = []
    skipped = 0
    for i, path in enumerate(json_files, 1):
        try:
            with open(path, "r", encoding="utf-8") as f:
                match = json.load(f)
            rows, meta = first_innings_states(path.stem, match)
            if rows:
                all_rows.extend(rows)
                match_metas.append(meta)
            else:
                skipped += 1
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"Skipping {path.name}: {exc}")
        if i % 500 == 0:
            print(f"Parsed {i:,}/{len(json_files):,} files...")

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No usable men's T20I innings were parsed.")

    match_meta_df = pd.DataFrame(match_metas).drop_duplicates("match_id")
    city_counts = match_meta_df["city"].value_counts()
    common_cities = set(city_counts[city_counts >= MIN_CITY_MATCHES].index)
    df["city"] = df["city"].where(df["city"].isin(common_cities), "Other")

    print(f"Usable matches: {df['match_id'].nunique():,}")
    print(f"Training-state rows: {len(df):,}")
    print(f"Teams: {df['batting_team'].nunique():,}")
    print(f"Cities after rare-city grouping: {df['city'].nunique():,}")
    return df


def build_pipeline(seed: int = RANDOM_STATE) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    model = XGBRegressor(
        n_estimators=450,
        max_depth=6,
        learning_rate=0.045,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def metrics(y_true, pred):
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": float(r2_score(y_true, pred)),
    }


def main():
    df = load_dataset()

    # Match-level split prevents states from the same match leaking into train and test.
    match_ids = np.array(sorted(df["match_id"].unique()))
    train_ids, test_ids = train_test_split(
        match_ids, test_size=0.20, random_state=RANDOM_STATE
    )
    train_df = df[df["match_id"].isin(train_ids)].copy()
    test_df = df[df["match_id"].isin(test_ids)].copy()

    pipe = build_pipeline()
    print("Training overall model...")
    pipe.fit(train_df[FEATURES], train_df["remaining_runs"])

    pred = np.maximum(0, pipe.predict(test_df[FEATURES]))
    overall = metrics(test_df["remaining_runs"], pred)

    late_train = train_df[train_df["balls_left"] <= 30].copy()
    late_test = test_df[test_df["balls_left"] <= 30].copy()
    late_pipe = build_pipeline(seed=RANDOM_STATE + 1)
    print("Training late-innings model...")
    late_pipe.fit(late_train[FEATURES], late_train["remaining_runs"])
    late_pred = np.maximum(0, late_pipe.predict(late_test[FEATURES]))
    late = metrics(late_test["remaining_runs"], late_pred)

    state_medians = (
        train_df.groupby(["balls_left", "wickets_left"])["remaining_runs"].median().to_dict()
    )
    ball_medians = train_df.groupby("balls_left")["remaining_runs"].median().to_dict()
    state_stats = {"state_medians": state_medians, "ball_medians": ball_medians}

    meta = {
        "source": "Cricsheet",
        "source_url": "https://cricsheet.org/downloads/",
        "archive_url": "https://cricsheet.org/downloads/t20s_json.zip",
        "data_format": "Cricsheet JSON",
        "gender_filter": "male",
        "target": "remaining_runs",
        "features": FEATURES,
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "matches": int(df["match_id"].nunique()),
        "train_matches": int(train_df["match_id"].nunique()),
        "test_matches": int(test_df["match_id"].nunique()),
        "test_mae": overall["mae"],
        "test_rmse": overall["rmse"],
        "test_r2": overall["r2"],
        "late_5_over_mae": late["mae"],
        "late_5_over_rmse": late["rmse"],
        "late_5_over_r2": late["r2"],
        "teams": sorted(df["batting_team"].unique().tolist()),
        "cities": sorted(df["city"].unique().tolist()),
        "random_state": RANDOM_STATE,
        "split": "match-level 80/20 holdout",
    }

    with open(ARTIFACT_DIR / "pipe.pkl", "wb") as f:
        pickle.dump(pipe, f)
    with open(ARTIFACT_DIR / "late_pipe.pkl", "wb") as f:
        pickle.dump(late_pipe, f)
    with open(ARTIFACT_DIR / "state_stats.pkl", "wb") as f:
        pickle.dump(state_stats, f)
    with open(ARTIFACT_DIR / "model_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    pd.DataFrame([{**overall, "split": "overall"}, {**late, "split": "last_5_overs"}]).to_csv(
        ARTIFACT_DIR / "metrics.csv", index=False
    )

    print("\nHeld-out test metrics")
    print(f"MAE : {overall['mae']:.2f} runs")
    print(f"RMSE: {overall['rmse']:.2f} runs")
    print(f"R²  : {overall['r2']:.3f}")
    print(f"Late-innings MAE: {late['mae']:.2f} runs")
    print(f"\nSaved model artifacts to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
