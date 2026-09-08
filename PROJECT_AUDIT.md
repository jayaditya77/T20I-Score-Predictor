# Final project audit

This audit was performed against the Cricsheet edition before the interview-ready packaging.

## Fixed

1. **Invalid current-over ball state** — the app now rejects 6+ legal balls in the current over. Six legal balls are represented as the next completed over with 0 balls.
2. **Stale Streamlit widget values** — prediction inputs are submitted through a form and validated again before any model calculation.
3. **Recent-window inconsistency** — runs in the recent 5-over window cannot exceed the current score; before five legal overs, the recent window must equal the innings score.
4. **End-of-innings handling** — at 20 overs or 10 wickets, the app returns the current score instead of sending an impossible state to XGBoost.
5. **False 6-runs-per-ball ceiling** — removed. Legal-ball count is not a strict mathematical run ceiling because wides, no-balls and overthrows can add runs without consuming a legal ball in the usual way.
6. **Unusually high run-rate diagnostic** — the app warns about extreme current run rates without rejecting unusual but possible scores.
7. **Model-loading failure handling** — corrupted/incompatible artifacts now produce a clear error instead of an opaque traceback.
8. **Download repeatability** — the Cricsheet downloader skips a download when the JSON snapshot is already extracted.
9. **Archive extraction safety** — ZIP members are checked before extraction to prevent path traversal.
10. **Shared feature logic** — innings phase calculation is now shared between training and the Streamlit app through `utils.py`.
11. **Validation tests** — `test_validation.py` covers legal-over boundaries, early-innings recent-window logic, innings arithmetic and phase boundaries.
12. **Reproducibility** — package versions are pinned to the versions used for the trained development run.
13. **Documentation** — README explains the input convention, model design, validation strategy, source, and interview explanation.

## Intentional design choices

- Prediction starts at 2 completed legal overs because the training data starts at 12 legal balls.
- The target is remaining runs, which keeps the model focused on the information available at prediction time.
- A match-level 80/20 split is used to reduce leakage from multiple states belonging to the same match.
- A separate model is used for the final five overs because late-innings dynamics differ from earlier phases.

## Verified locally

- Python syntax compilation passed for all project modules.
- All validation tests passed.
- The original Cricsheet training run produced 3,319 usable matches and the recorded held-out metrics shown by the app: MAE 15.16, RMSE 20.89, R² 0.806, late-innings MAE 6.87.
