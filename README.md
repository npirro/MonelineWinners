# Moneyline Winners — Streamlit App v3

This version adds optional odds display while keeping odds out of the prediction model.

## Important

Moneyline odds DO NOT affect:
- Win Score
- Rank
- Grade
- Suggested Status

Odds are only displayed so the user can decide whether the reward is worth betting.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Columns

Required:
Game,Team,Opponent,Home,SP_Score,Offense_Score,Bullpen_Score,Lineup_Score,Situational_Score,Notes

Optional:
Moneyline
