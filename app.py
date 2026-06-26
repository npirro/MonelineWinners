import os
import re
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================
# App Config
# =========================

st.set_page_config(
    page_title="Moneyline Winners v1.0",
    page_icon="⚾",
    layout="wide",
)

MLB_BASE = "https://statsapi.mlb.com/api/v1"
MODEL_PATH = Path("model_artifacts/mlb_logistic_model.joblib")

DEFAULT_FEATURES = [
    "win_pct_diff",
    "rpg_diff",
    "rapg_diff",
    "run_diff_per_game_diff",
    "recent_win_pct_diff",
    "recent_rpg_diff",
    "recent_rapg_diff",
    "home_field",
]

FINAL_STATES = {"Final", "Game Over", "Completed Early"}


# =========================
# Styling
# =========================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #07111f;
            color: #eaf2ff;
        }

        h1, h2, h3 {
            color: #eaf2ff;
        }

        .block-container {
            padding-top: 2rem;
        }

        .metric-card {
            background: linear-gradient(135deg, #0d1b2e, #102944);
            border: 1px solid #1f456e;
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 12px;
            box-shadow: 0 0 18px rgba(0,0,0,0.25);
        }

        .big-number {
            font-size: 34px;
            font-weight: 800;
            color: #8fd3ff;
        }

        .small-label {
            font-size: 13px;
            color: #b8c7d9;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .tier-a {
            color: #7CFFB2;
            font-weight: 800;
        }

        .tier-b {
            color: #FFE27A;
            font-weight: 800;
        }

        .pass {
            color: #ff9b9b;
            font-weight: 800;
        }

        div[data-testid="stDataFrame"] {
            background-color: #0d1b2e;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Helpers
# =========================

def normalize_team_name(name):
    if not name:
        return ""
    name = name.lower()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def american_to_implied_prob(moneyline):
    if moneyline is None or pd.isna(moneyline):
        return None

    moneyline = float(moneyline)

    if moneyline < 0:
        return abs(moneyline) / (abs(moneyline) + 100)

    return 100 / (moneyline + 100)


def prob_to_fair_moneyline(prob):
    prob = float(prob)

    if prob <= 0 or prob >= 1:
        return None

    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))

    return int(round(100 * (1 - prob) / prob))


def format_moneyline(x):
    if x is None or pd.isna(x):
        return "—"

    x = int(round(float(x)))

    if x > 0:
        return f"+{x}"

    return str(x)


def format_percent(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.1%}"


def get_secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def regular_season_start_for_year(year):
    return date(year, 3, 1)


# =========================
# Model Loading
# =========================

@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at: {MODEL_PATH}\n\n"
            "Make sure mlb_logistic_model.joblib is inside model_artifacts/"
        )
        st.stop()

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("pipeline") or artifact.get("estimator")
        features = (
            artifact.get("features")
            or artifact.get("feature_names")
            or artifact.get("selected_features")
        )

        if model is None:
            st.error("Loaded model artifact is a dictionary, but no model was found inside it.")
            st.stop()

        if features is None:
            features = infer_model_features(model)

        return model, list(features)

    model = artifact
    features = infer_model_features(model)

    return model, features


def infer_model_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return DEFAULT_FEATURES.copy()


# =========================
# MLB Data
# =========================

@st.cache_data(ttl=60 * 30)
def fetch_schedule_range(start_day, end_day):
    params = {
        "sportId": 1,
        "startDate": start_day,
        "endDate": end_day,
        "gameTypes": "R",
        "hydrate": "team,linescore,probablePitcher",
    }

    r = requests.get(f"{MLB_BASE}/schedule", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def flatten_games(schedule_json):
    games = []

    for date_block in schedule_json.get("dates", []):
        for game in date_block.get("games", []):
            if game.get("gameType") != "R":
                continue
            games.append(game)

    return games


def is_final_game(game):
    status = game.get("status", {}).get("detailedState", "")
    teams = game.get("teams", {})
    home_score = teams.get("home", {}).get("score")
    away_score = teams.get("away", {}).get("score")

    return status in FINAL_STATES and home_score is not None and away_score is not None


def create_team_state():
    return {
        "games": 0,
        "wins": 0,
        "runs_for": 0,
        "runs_against": 0,
        "recent": deque(maxlen=10),
    }


def update_team_state(state, runs_for, runs_against):
    win = 1 if runs_for > runs_against else 0

    state["games"] += 1
    state["wins"] += win
    state["runs_for"] += runs_for
    state["runs_against"] += runs_against

    state["recent"].append(
        {
            "win": win,
            "runs_for": runs_for,
            "runs_against": runs_against,
        }
    )


def snapshot_team_state(state):
    games = state["games"]

    if games <= 0:
        return {
            "win_pct": 0.500,
            "rpg": 4.50,
            "rapg": 4.50,
            "run_diff_per_game": 0.00,
            "recent_win_pct": 0.500,
            "recent_rpg": 4.50,
            "recent_rapg": 4.50,
        }

    win_pct = state["wins"] / games
    rpg = state["runs_for"] / games
    rapg = state["runs_against"] / games
    run_diff_per_game = rpg - rapg

    recent = list(state["recent"])

    if len(recent) == 0:
        recent_win_pct = win_pct
        recent_rpg = rpg
        recent_rapg = rapg
    else:
        recent_win_pct = sum(g["win"] for g in recent) / len(recent)
        recent_rpg = sum(g["runs_for"] for g in recent) / len(recent)
        recent_rapg = sum(g["runs_against"] for g in recent) / len(recent)

    return {
        "win_pct": win_pct,
        "rpg": rpg,
        "rapg": rapg,
        "run_diff_per_game": run_diff_per_game,
        "recent_win_pct": recent_win_pct,
        "recent_rpg": recent_rpg,
        "recent_rapg": recent_rapg,
    }


def build_team_states(history_games):
    team_states = defaultdict(create_team_state)

    final_games = [g for g in history_games if is_final_game(g)]

    final_games = sorted(
        final_games,
        key=lambda g: (
            g.get("officialDate", ""),
            g.get("gamePk", 0),
        ),
    )

    for game in final_games:
        teams = game.get("teams", {})

        home = teams.get("home", {})
        away = teams.get("away", {})

        home_team = home.get("team", {})
        away_team = away.get("team", {})

        home_id = home_team.get("id")
        away_id = away_team.get("id")

        home_score = int(home.get("score", 0))
        away_score = int(away.get("score", 0))

        update_team_state(team_states[home_id], home_score, away_score)
        update_team_state(team_states[away_id], away_score, home_score)

    return team_states


def build_features_for_game(game, team_states):
    teams = game.get("teams", {})

    home = teams.get("home", {})
    away = teams.get("away", {})

    home_team = home.get("team", {})
    away_team = away.get("team", {})

    home_id = home_team.get("id")
    away_id = away_team.get("id")

    home_snapshot = snapshot_team_state(team_states[home_id])
    away_snapshot = snapshot_team_state(team_states[away_id])

    features = {
        "win_pct_diff": home_snapshot["win_pct"] - away_snapshot["win_pct"],
        "rpg_diff": home_snapshot["rpg"] - away_snapshot["rpg"],

        # Positive means the away team has allowed more runs per game than the home team.
        "rapg_diff": away_snapshot["rapg"] - home_snapshot["rapg"],

        "run_diff_per_game_diff": (
            home_snapshot["run_diff_per_game"] - away_snapshot["run_diff_per_game"]
        ),
        "recent_win_pct_diff": (
            home_snapshot["recent_win_pct"] - away_snapshot["recent_win_pct"]
        ),
        "recent_rpg_diff": home_snapshot["recent_rpg"] - away_snapshot["recent_rpg"],

        # Keep this as home recent runs allowed minus away recent runs allowed.
        # This matches the research engine's observed negative coefficient behavior.
        "recent_rapg_diff": (
            home_snapshot["recent_rapg"] - away_snapshot["recent_rapg"]
        ),

        "home_field": 1.0,
    }

    return features


def get_probable_pitcher(game, side):
    try:
        pitcher = game.get("teams", {}).get(side, {}).get("probablePitcher")
        if pitcher:
            return pitcher.get("fullName", "TBD")
    except Exception:
        pass

    return "TBD"


# =========================
# Odds
# =========================

@st.cache_data(ttl=60 * 10)
def fetch_moneyline_odds(api_key):
    if not api_key:
        return {}

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    data = r.json()
    odds_by_team = {}

    for event in data:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    team_name = outcome.get("name")
                    price = outcome.get("price")

                    if team_name is None or price is None:
                        continue

                    key = normalize_team_name(team_name)

                    # Higher American price is always better for the bettor:
                    # +135 beats +120, and -105 beats -120.
                    if key not in odds_by_team or price > odds_by_team[key]:
                        odds_by_team[key] = price

    return odds_by_team


# =========================
# Grading
# =========================

def grade_edge(edge):
    if edge is None or pd.isna(edge):
        return "MODEL"

    if edge >= 0.05:
        return "A"

    if edge >= 0.035:
        return "B"

    if edge >= 0.02:
        return "C"

    return "PASS"


def grade_sort_value(grade):
    order = {
        "A": 4,
        "B": 3,
        "C": 2,
        "MODEL": 1,
        "PASS": 0,
    }
    return order.get(grade, 0)


# =========================
# App
# =========================

st.title("⚾ Moneyline Winners v1.0")
st.caption("Production Model v1 — validated team-only logistic model. Numbers decide. Not opinions.")

model, model_features = load_model_artifact()

today_et = datetime.now(ZoneInfo("America/New_York")).date()

with st.sidebar:
    st.header("Controls")

    selected_date = st.date_input(
        "Game Date",
        value=today_et,
    )

    default_odds_key = os.getenv("ODDS_API_KEY", "") or get_secret("ODDS_API_KEY")

    odds_api_key = st.text_input(
        "Odds API Key",
        value=default_odds_key,
        type="password",
        help="Optional. Leave blank to show model probabilities without market edge.",
    )

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.write("### Model")
    st.write("Production Feature Set:")
    st.code(", ".join(model_features), language="text")

    st.write("### Notes")
    st.write(
        "v1.0 uses the trained research model. Odds are used only after prediction "
        "to calculate market edge."
    )


selected_date = pd.to_datetime(selected_date).date()
season_start = regular_season_start_for_year(selected_date.year)
history_end = selected_date - timedelta(days=1)

try:
    if history_end >= season_start:
        history_json = fetch_schedule_range(
            season_start.isoformat(),
            history_end.isoformat(),
        )
        history_games = flatten_games(history_json)
    else:
        history_games = []

    today_json = fetch_schedule_range(
        selected_date.isoformat(),
        selected_date.isoformat(),
    )
    today_games = flatten_games(today_json)

except Exception as e:
    st.error(f"MLB data download failed: {e}")
    st.stop()


team_states = build_team_states(history_games)

try:
    odds_by_team = fetch_moneyline_odds(odds_api_key)
except Exception as e:
    odds_by_team = {}
    st.warning(f"Odds download failed. Showing model-only board. Error: {e}")


if not today_games:
    st.warning("No regular-season MLB games found for the selected date.")
    st.stop()


game_rows = []
all_side_rows = []

for game in today_games:
    teams = game.get("teams", {})

    home_team = teams.get("home", {}).get("team", {})
    away_team = teams.get("away", {}).get("team", {})

    home_name = home_team.get("name", "Home")
    away_name = away_team.get("name", "Away")

    home_abbr = home_team.get("abbreviation") or home_team.get("teamName") or home_name
    away_abbr = away_team.get("abbreviation") or away_team.get("teamName") or away_name

    game_label = f"{away_abbr} @ {home_abbr}"

    features = build_features_for_game(game, team_states)

    X = pd.DataFrame([features])

    for f in model_features:
        if f not in X.columns:
            X[f] = 0.0

    X = X[model_features].astype(float)

    try:
        home_prob = float(model.predict_proba(X)[0][1])
    except Exception as e:
        st.error(f"Prediction failed for {game_label}: {e}")
        st.stop()

    away_prob = 1 - home_prob

    home_key = normalize_team_name(home_name)
    away_key = normalize_team_name(away_name)

    home_market_ml = odds_by_team.get(home_key)
    away_market_ml = odds_by_team.get(away_key)

    home_market_prob = american_to_implied_prob(home_market_ml)
    away_market_prob = american_to_implied_prob(away_market_ml)

    home_edge = None if home_market_prob is None else home_prob - home_market_prob
    away_edge = None if away_market_prob is None else away_prob - away_market_prob

    candidates = [
        {
            "game": game_label,
            "team": home_abbr,
            "team_full": home_name,
            "opponent": away_abbr,
            "side": "Home",
            "model_prob": home_prob,
            "fair_ml": prob_to_fair_moneyline(home_prob),
            "market_ml": home_market_ml,
            "market_prob": home_market_prob,
            "edge": home_edge,
            "probable_pitcher": get_probable_pitcher(game, "home"),
        },
        {
            "game": game_label,
            "team": away_abbr,
            "team_full": away_name,
            "opponent": home_abbr,
            "side": "Away",
            "model_prob": away_prob,
            "fair_ml": prob_to_fair_moneyline(away_prob),
            "market_ml": away_market_ml,
            "market_prob": away_market_prob,
            "edge": away_edge,
            "probable_pitcher": get_probable_pitcher(game, "away"),
        },
    ]

    all_side_rows.extend(candidates)

    priced_candidates = [c for c in candidates if c["edge"] is not None]

    if priced_candidates:
        best = max(priced_candidates, key=lambda c: c["edge"])

        if best["edge"] <= 0:
            best = max(candidates, key=lambda c: c["model_prob"])
    else:
        best = max(candidates, key=lambda c: c["model_prob"])

    best["grade"] = grade_edge(best["edge"])

    game_rows.append(best)


board = pd.DataFrame(game_rows)
all_sides = pd.DataFrame(all_side_rows)

board["grade_sort"] = board["grade"].apply(grade_sort_value)

if board["edge"].notna().any():
    board = board.sort_values(
        ["grade_sort", "edge", "model_prob"],
        ascending=[False, False, False],
    )
else:
    board = board.sort_values("model_prob", ascending=False)


# =========================
# KPI Cards
# =========================

total_games = len(today_games)
a_count = int((board["grade"] == "A").sum())
positive_edges = int((board["edge"].fillna(-1) > 0).sum())
top_model_prob = float(board["model_prob"].max())

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">Games Scored</div>
            <div class="big-number">{total_games}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">Tier A Candidates</div>
            <div class="big-number">{a_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">Positive Edges</div>
            <div class="big-number">{positive_edges}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">Top Model Probability</div>
            <div class="big-number">{top_model_prob:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Display Board
# =========================

st.subheader("Moneyline Board")

display_board = board.copy()

display_board["Model Prob"] = display_board["model_prob"].apply(format_percent)
display_board["Fair ML"] = display_board["fair_ml"].apply(format_moneyline)
display_board["Market ML"] = display_board["market_ml"].apply(format_moneyline)
display_board["Market Implied"] = display_board["market_prob"].apply(format_percent)
display_board["Edge"] = display_board["edge"].apply(format_percent)

display_board = display_board[
    [
        "grade",
        "game",
        "team",
        "side",
        "opponent",
        "Model Prob",
        "Fair ML",
        "Market ML",
        "Market Implied",
        "Edge",
        "probable_pitcher",
    ]
].rename(
    columns={
        "grade": "Tier",
        "game": "Game",
        "team": "Model Side",
        "side": "Home/Away",
        "opponent": "Opponent",
        "probable_pitcher": "Probable Pitcher",
    }
)

st.dataframe(
    display_board,
    use_container_width=True,
    hide_index=True,
)


with st.expander("Show both sides for every game"):
    side_display = all_sides.copy()
    side_display["Grade"] = side_display["edge"].apply(grade_edge)
    side_display["Model Prob"] = side_display["model_prob"].apply(format_percent)
    side_display["Fair ML"] = side_display["fair_ml"].apply(format_moneyline)
    side_display["Market ML"] = side_display["market_ml"].apply(format_moneyline)
    side_display["Market Implied"] = side_display["market_prob"].apply(format_percent)
    side_display["Edge"] = side_display["edge"].apply(format_percent)

    side_display = side_display[
        [
            "Grade",
            "game",
            "team",
            "side",
            "opponent",
            "Model Prob",
            "Fair ML",
            "Market ML",
            "Market Implied",
            "Edge",
            "probable_pitcher",
        ]
    ].rename(
        columns={
            "game": "Game",
            "team": "Team",
            "side": "Home/Away",
            "opponent": "Opponent",
            "probable_pitcher": "Probable Pitcher",
        }
    )

    st.dataframe(
        side_display,
        use_container_width=True,
        hide_index=True,
    )


st.divider()

st.caption(
    "Important: v1.0 uses the validated production research model. "
    "Market odds are not used to create the prediction; odds are only used afterward "
    "to compare price vs probability."
)