from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

from utils import innings_state, phase_name, validate_match_state

BASE = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE / "artifacts"

st.set_page_config(
    page_title="T20I Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_ARTIFACTS = ["pipe.pkl", "late_pipe.pkl", "state_stats.pkl", "model_meta.pkl"]
missing = [name for name in REQUIRED_ARTIFACTS if not (ARTIFACT_DIR / name).exists()]

if missing:
    st.title("🏏 T20I Score Predictor")
    st.error("Model artifacts are missing. Run the training setup once before starting the app.")
    st.code("python3.11 setup_cricsheet.py\npython3.11 -m streamlit run app.py", language="bash")
    st.caption("The setup script downloads the official Cricsheet T20I JSON archive and trains the models locally.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(ARTIFACT_DIR / "pipe.pkl", "rb") as f:
        pipe = pickle.load(f)
    with open(ARTIFACT_DIR / "late_pipe.pkl", "rb") as f:
        late_pipe = pickle.load(f)
    with open(ARTIFACT_DIR / "state_stats.pkl", "rb") as f:
        state_stats = pickle.load(f)
    with open(ARTIFACT_DIR / "model_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return pipe, late_pipe, state_stats, meta


try:
    pipe, late_pipe, state_stats, meta = load_artifacts()
except Exception as exc:
    st.title("🏏 T20I Score Predictor")
    st.error("The saved model artifacts could not be loaded with the installed Python packages.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.caption("Use the versions in requirements.txt, then rerun setup_cricsheet.py if necessary.")
    st.stop()

teams = list(meta.get("teams", []))
cities = list(meta.get("cities", []))

st.set_page_config(
    page_title="T20I Score Predictor",
    page_icon="🏏",
    layout="wide",
)

st.title("T20I Score Predictor")
st.write("Predict the final score of a T20I first innings.")

st.sidebar.header("Model Information")
st.sidebar.write(f"Test MAE: {meta['test_mae']:.1f} runs")
st.sidebar.write(f"Test RMSE: {meta['test_rmse']:.1f} runs")
st.sidebar.write(f"Test R²: {meta['test_r2']:.3f}")
st.sidebar.write(
    f"Late 5-over MAE: {meta['late_5_over_mae']:.1f} runs"
)

st.sidebar.write(f"Training rows: {meta['training_rows']:,}")
st.sidebar.write(
    f"Train/test matches: "
    f"{meta['train_matches']:,} / {meta['test_matches']:,}"
)

st.sidebar.divider()
st.sidebar.write("Dataset: Cricsheet")
st.sidebar.write("Format: T20I JSON")
st.sidebar.write("Filter: Men's T20 internationals")

with st.expander("How the model works"):
    st.write(
        "The model predicts runs remaining and adds them to the current score."
    )
    st.write(
        "It uses match state features such as score, balls remaining, "
        "wickets remaining, run rate, recent scoring rate, teams and venue."
    )
    st.write(
        "A separate XGBoost model is used during the final five overs."
    )

st.header("Match Situation")

with st.form("prediction_form"):

    batting_team = st.selectbox(
        "Batting team",
        teams,
        key="batting_team",
    )

    bowling_options = [
        team for team in teams if team != batting_team
    ]

    bowling_team = st.selectbox(
        "Bowling team",
        bowling_options,
        key="bowling_team",
    )

    city_default = (
        cities.index("Other")
        if "Other" in cities
        else 0
    )

    city = st.selectbox(
        "City / venue region",
        cities,
        index=city_default,
    )

    current_score = st.number_input(
        "Current score",
        min_value=0,
        max_value=500,
        value=80,
        step=1,
    )

    completed_overs = st.number_input(
        "Completed overs",
        min_value=2,
        max_value=20,
        value=10,
        step=1,
        help="The model is trained from 2 legal overs onward.",
    )

    balls_in_over = st.number_input(
        "Legal balls completed in current over",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        help="Enter 0–5 legal balls.",
    )

    wickets = st.number_input(
        "Wickets lost",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
    )

    last_five = st.number_input(
        "Runs in last 5 overs",
        min_value=0,
        max_value=500,
        value=40,
        step=1,
    )

    submitted = st.form_submit_button(
        "Predict Final Score",
        type="primary",
    )


if submitted:

    balls_in_over = int(balls_in_over)
    completed_overs = int(completed_overs)
    current_score = int(current_score)
    wickets = int(wickets)
    last_five = int(last_five)

    errors = validate_match_state(
        current_score,
        completed_overs,
        balls_in_over,
        wickets,
        last_five,
    )

    balls_bowled = (
        completed_overs * 6
        + balls_in_over
    )

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    balls_left, wickets_left = innings_state(
        completed_overs,
        balls_in_over,
        wickets,
    )

    if balls_bowled > 0:
        crr_input = (
            current_score * 6.0
            / balls_bowled
        )

        if crr_input > 20:
            st.warning(
                f"Current run rate is {crr_input:.1f} runs/over."
            )

    if balls_left == 0 or wickets_left == 0:

        predicted_score = current_score
        predicted_remaining = 0.0
        model_name = "Completed-innings rule"

    else:

        crr = (
            current_score * 6.0
            / balls_bowled
        )

        lookback_balls = min(
            30,
            balls_bowled,
        )

        recent_rr = (
            last_five * 6.0
            / lookback_balls
        )

        baseline_rr = (
            0.65 * recent_rr
            + 0.35 * crr
        )

        baseline_remaining = (
            baseline_rr
            * balls_left
            / 6.0
        )

        recent_projected_final = (
            current_score
            + recent_rr * balls_left / 6.0
        )

        recent_gap = recent_rr - crr

        phase = phase_name(
            balls_bowled
        )

        X = pd.DataFrame(
            {
                "batting_team": [batting_team],
                "bowling_team": [bowling_team],
                "city": [city],
                "phase": [phase],
                "current_score": [current_score],
                "balls_left": [balls_left],
                "wickets_left": [wickets_left],
                "crr": [crr],
                "last_five": [last_five],
                "recent_rr": [recent_rr],
                "baseline_rr": [baseline_rr],
                "baseline_remaining": [
                    baseline_remaining
                ],
                "recent_projected_final": [
                    recent_projected_final
                ],
                "recent_gap": [recent_gap],
            }
        )

        active = (
            late_pipe
            if balls_left <= 30
            else pipe
        )

        raw_remaining = max(
            0.0,
            float(active.predict(X)[0]),
        )

        predicted_remaining = raw_remaining

        predicted_score = max(
            current_score,
            int(
                round(
                    current_score
                    + predicted_remaining
                )
            ),
        )

        model_name = (
            "Late-innings XGBoost"
            if balls_left <= 30
            else "XGBoost"
        )

    st.divider()
    st.subheader("Prediction")

    st.metric(
        "Predicted Final Score",
        predicted_score,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Runs Remaining",
            int(
                round(
                    predicted_score
                    - current_score
                )
            ),
        )

    with col2:
        st.metric(
            "Balls Remaining",
            balls_left,
        )

    with col3:
        st.write("Model")
        st.write(model_name)

    mae = float(
        meta["late_5_over_mae"]
        if balls_left <= 30
        else meta["test_mae"]
    )

    low = max(
        current_score,
        int(
            round(
                predicted_score - mae
            )
        ),
    )

    high = int(
        round(
            predicted_score + mae
        )
    )

    st.info(
        f"Approximate prediction range: {low}–{high} runs "
        f"(based on a held-out MAE of about ±{mae:.1f} runs)."
    )

    progress = max(
        0.0,
        min(
            1.0,
            balls_bowled / 120.0,
        ),
    )

    st.progress(
        progress,
        text=(
            f"Innings progress: {balls_bowled}/120 legal balls "
            f"({progress:.0%})"
        ),
    )

    if balls_left <= 30:
        st.caption("Late-innings model active: final 5 overs.")
    else:
        st.caption("Overall XGBoost model active.")
