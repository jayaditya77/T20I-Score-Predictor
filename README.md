T20I Score Predictor

A machine learning project that predicts the final score of a T20I first innings using ball-by-ball match data.

Tech Stack
- Python
- XGBoost
- Scikit-learn
- Pandas
- Streamlit

Features
- Ball-by-ball feature engineering
- XGBoost score prediction
- Separate model for the final 5 overs
- Streamlit web application

Results
- R²: 0.806
- MAE: 15.16 runs
- Late-innings MAE: 6.87 runs

Dataset
Data sourced from [Cricsheet](https://cricsheet.org/).

```bash
pip install -r requirements.txt
python3.11 -m streamlit run app.py
