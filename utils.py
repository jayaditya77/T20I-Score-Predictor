"""Pure cricket-state helpers shared by the Streamlit app."""


def validate_match_state(
    current_score: int,
    completed_overs: int,
    balls_in_over: int,
    wickets: int,
    last_five: int,
) -> list[str]:
    """Return validation errors for a live first-innings T20 state."""
    errors = []

    if not 0 <= balls_in_over <= 5:
        errors.append("Legal balls completed in the current over must be between 0 and 5.")
    if not 2 <= completed_overs <= 20:
        errors.append("Completed overs must be between 2 and 20.")
    if completed_overs == 20 and balls_in_over != 0:
        errors.append("At 20 completed overs, the current-over ball count must be 0.")
    if not 0 <= wickets <= 10:
        errors.append("Wickets lost must be between 0 and 10.")
    if current_score < 0:
        errors.append("Current score cannot be negative.")
    if last_five < 0:
        errors.append("Runs in the last 5 overs cannot be negative.")
    if last_five > current_score:
        errors.append("Runs in the last 5 overs cannot exceed the current innings score.")

    balls_bowled = completed_overs * 6 + balls_in_over
    if balls_bowled > 120:
        errors.append("A regulation T20 innings cannot exceed 120 legal balls.")

    if balls_bowled < 30 and last_five != current_score:
        errors.append(
            "With fewer than 5 completed legal overs, 'runs in last 5 overs' must equal the current score."
        )

    return errors


def innings_state(completed_overs: int, balls_in_over: int, wickets: int) -> tuple[int, int]:
    """Return legal balls bowled and wickets remaining."""
    balls_bowled = completed_overs * 6 + balls_in_over
    balls_left = 120 - balls_bowled
    wickets_left = 10 - wickets
    return balls_left, wickets_left


def phase_name(balls_bowled: int) -> str:
    if balls_bowled <= 36:
        return "powerplay"
    if balls_bowled <= 72:
        return "middle"
    if balls_bowled <= 96:
        return "build"
    return "death"
