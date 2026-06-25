# Moneyline Winners — Streamlit App v11 Phase 2

Phase 2 upgrade:
- Replaces Model Confidence with Projected Win Probability
- Ranks suggested sides by Projected Win Probability
- Calculates head-to-head matchup advantages:
  - SP Advantage
  - Offense Advantage
  - Bullpen/Pitching Advantage
  - Lineup Advantage
  - Situational Advantage
- Converts matchup edge into projected win probability using a first-pass logistic curve
- Keeps one-side-per-game logic
- Keeps odds informational only

Important:
This is a first-pass probability model. It needs backtesting/calibration next.

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```
