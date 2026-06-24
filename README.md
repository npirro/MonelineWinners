# Moneyline Winners

Standalone MLB projected winner dashboard.

## Purpose
Ranks MLB teams by projected likelihood of winning based on baseball metrics only.

This app does NOT use:
- Sportsbook odds
- Implied probability
- EV
- Parlay logic

## Daily Workflow
1. Get today's MLB team metrics.
2. Paste CSV rows into the app or upload a CSV file.
3. Click Refresh Rankings.
4. Review A/B/C/Pass grades.
5. Export the ranked board if desired.

## Required CSV Columns
Game,Team,Opponent,Home,SP_Score,Offense_Score,Bullpen_Score,Lineup_Score,Situational_Score,Notes

## Default Weights
- Starting Pitcher: 40
- Offense: 25
- Bullpen: 15
- Confirmed Lineup: 12
- Situational: 8

## Grade Rules
- 85+ = A
- 78-84 = B
- 70-77 = C
- Below 70 = Pass

Open index.html in your browser.
