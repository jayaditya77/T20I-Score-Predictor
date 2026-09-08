from utils import innings_state, phase_name, validate_match_state


def check(condition, message):
    if not condition:
        raise AssertionError(message)


# Legal over-state boundaries.
check(validate_match_state(80, 10, 0, 2, 40) == [], "10.0 should be valid")
check(validate_match_state(80, 10, 5, 2, 40) == [], "10.5 should be valid")
check(any("between 0 and 5" in e for e in validate_match_state(80, 10, 6, 2, 40)), "10.6 must be rejected")
check(any("must be 0" in e for e in validate_match_state(160, 20, 1, 2, 40)), "20.1 must be rejected")

# Recent-window consistency.
check(any("must equal the current score" in e for e in validate_match_state(80, 3, 0, 2, 30)), "Early innings recent score mismatch must be rejected")
check(validate_match_state(80, 10, 0, 2, 40) == [], "Normal state must be valid")
check(any("cannot exceed" in e for e in validate_match_state(80, 10, 0, 2, 81)), "Recent runs > innings score must be rejected")

# State arithmetic.
check(innings_state(10, 0, 2) == (60, 8), "10.0 should leave 60 balls and 8 wickets")
check(innings_state(10, 5, 2) == (55, 8), "10.5 should leave 55 balls and 8 wickets")
check(phase_name(36) == "powerplay", "36 legal balls should be powerplay")
check(phase_name(37) == "middle", "37 legal balls should be middle")
check(phase_name(96) == "build", "96 legal balls should be build")
check(phase_name(97) == "death", "97 legal balls should be death")

print("All validation tests passed.")
