# Moneyline Winners — Streamlit App

A Python/Streamlit MLB projected winners dashboard.

## Purpose

Ranks MLB teams by projected likelihood of winning based on baseball metrics only.

This app does NOT use:
- Sportsbook odds
- Implied probability
- EV
- Parlay logic

## Files

- `app.py` — Streamlit app
- `requirements.txt` — Python dependencies

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Daily Workflow

1. Open the app.
2. Upload a CSV or edit the sample slate.
3. Review the ranked projected winners board.
4. Use A/B grades as the strongest projected winner shortlist.
5. Export the ranked board.

## Required CSV Columns

Game,Team,Opponent,Home,SP_Score,Offense_Score,Bullpen_Score,Lineup_Score,Situational_Score,Notes

## Default Weights

- Starting Pitcher: 40
- Offense: 25
- Bullpen: 15
- Confirmed Lineup: 12
- Situational: 8
