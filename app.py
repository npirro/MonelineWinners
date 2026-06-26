import os
import re
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import html as html_lib

import joblib
import pandas as pd
import requests
import streamlit as st


# =========================
# App Config
# =========================

st.set_page_config(
    page_title="Moneyline Winners v1.0.2",
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

PREGAME_STATES = {
    "Scheduled",
    "Pre-Game",
    "Warmup",
    "Delayed Start",
    "Delayed",
    "Postponed",
}


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
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .metric-card {
            background: linear-gradient(135deg, #0d1b2e, #102944);
            border: 1px solid #1f456e;
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 12px;
            box-shadow: 0 0 18px rgba(0,0,0,0.25);
        }

        .game-card {
            background: linear-gradient(135deg, #0d1b2e, #102944);
            border: 1px solid #1f456e;
            border-radius: 20px;
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: 0 0 18px rgba(0,0,0,0.28);
            min-height: 375px;
        }

        .game-title {
            font-size: 19px;
            font-weight: 800;
            color: #eaf2ff;
            margin-bottom: 5px;
        }

        .game-subtitle {
            font-size: 13px;
            color: #b8c7d9;
            margin-bottom: 14px;
        }

        .rank-badge {
            display: inline-block;
            color: #07111f;
            background: #8fd3ff;
            border-radius: 999px;
            padding: 3px 9px;
            font-size: 12px;
            font-weight: 900;
            margin-right: 8px;
            vertical-align: middle;
        }

        .model-side {
            font-size: 28px;
            font-weight: 900;
            color: #8fd3ff;
            margin-top: 4px;
            margin-bottom: 8px;
        }

        .model-prob {
            font-size: 38px;
            font-weight: 900;
            color: #7CFFB2;
            line-height: 1.0;
        }

        .small-label {
            font-size: 12px;
            color: #b8c7d9;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 2px;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 14px;
            margin-bottom: 12px;
        }

        .mini-box {
            background-color: rgba(255,255,255,0.045);
            border: 1px solid rgba(143,211,255,0.16);
            border-radius: 12px;
            padding: 10px;
            min-height: 62px;
        }

        .mini-value {
            font-size: 17px;
            font-weight: 800;
            color: #eaf2ff;
            overflow-wrap: anywhere;
        }

        .why-box {
            background-color: rgba(255,255,255,0.035);
            border: 1px solid rgba(143,211,255,0.12);
            border-radius: 12px;
            padding: 10px 12px;
            margin-top: 10px;
        }

        .why-line {
            color: #dceaff;
            font-size: 13px;
            margin-top: 4px;
            line-height: 1.35;
        }

        .status-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 12px;
        }

        .pill {
            display: inline-block;
            padding: 5px 9px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            border: 1px solid rgba(255,255,255,0.14);
        }

        .pill-good {
            color: #7CFFB2;
            background-color: rgba(124,255,178,0.10);
        }

        .pill-warn {
            color: #FFE27A;
            background-color: rgba(255,226,122,0.10);
        }

        .pill-bad {
            color: #ff9b9b;
            background-color: rgba(255,155,155,0.10);
        }

        .pill-info {
            color: #8fd3ff;
            background-color: rgba(143,211,255,0.10);
        }

        .big-number {
            font-size: 34px;
            font-weight: 800;
            color: #8fd3ff;
        }

        .muted {
            color: #b8c7d9;
            font-size: 13px;
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

def h(value):
    return html_lib.escape(str(value))


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


def get_game_status(game):
    return game.get("status", {}).get("detailedState", "")


def is_final_game(game):
    status = get_game_status(game)
    teams = game.get("teams", {})
    home_score = teams.get("home", {}).get("score")
    away_score = teams.get("away", {}).get("score")

    return status in FINAL_STATES and home_score is not None and away_score is not None


def is_pregame(game):
    status = get_game_status(game)
    return status in PREGAME_STATES or "Scheduled" in status or "Pre-Game" in status


def get_game_time_et(game):
    game_datetime = game.get("gameDate")

    if not game_datetime:
        return "TBD"

    try:
        dt = datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
        et = dt.astimezone(ZoneInfo("America/New_York"))

        try:
            return et.strftime("%-I:%M %p")
        except Exception:
            return et.strftime("%I:%M %p").lstrip("0")

    except Exception:
        return "TBD"


def grade_lineup_status(status):
    if status == "Confirmed":
        return "pill-good"
    if status == "Partial":
        return "pill-warn"
    return "pill-bad"


def grade_model_tier(prob):
    if prob >= 0.58:
        return "A"
    if prob >= 0.56:
        return "B"
    if prob >= 0.54:
        return "C"
    return "Lean"


def tier_class(model_tier):
    if model_tier == "A":
        return "pill-good"
    if model_tier in {"B", "C"}:
        return "pill-info"
    return "pill-warn"


def lineup_sort_value(status):
    order = {
        "Confirmed": 3,
        "Partial": 2,
        "Not Confirmed": 1,
        "Unknown": 0,
    }
    return order.get(status, 0)


def build_reasons(features, selected_side):
    reasons = []

    side_is_home = selected_side == "Home"

    def favors_selected(value, home_positive=True):
        if home_positive:
            return value > 0 if side_is_home else value < 0
        return value < 0 if side_is_home else value > 0

    checks = [
        ("Win rate edge", features.get("win_pct_diff", 0), 0.025, True),
        ("Run differential edge", features.get("run_diff_per_game_diff", 0), 0.15, True),
        ("Better scoring profile", features.get("rpg_diff", 0), 0.20, True),
        ("Better run prevention", features.get("rapg_diff", 0), 0.20, True),
        ("Recent form edge", features.get("recent_win_pct_diff", 0), 0.10, True),
        ("Recent offense edge", features.get("recent_rpg_diff", 0), 0.30, True),

        # recent_rapg_diff = home recent runs allowed - away recent runs allowed.
        # Lower is better for home, higher is better for away.
        ("Recent run prevention edge", features.get("recent_rapg_diff", 0), 0.30, False),
    ]

    for label, value, threshold, home_positive in checks:
        if abs(value) >= threshold and favors_selected(value, home_positive=home_positive):
            reasons.append(label)

    if not reasons:
        reasons.append("Narrow model edge")
        reasons.append("No major single-driver advantage")

    return reasons[:3]


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


@st.cache_data(ttl=60 * 5)
def fetch_boxscore(game_pk):
    r = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=20)
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
        "rapg_diff": away_snapshot["rapg"] - home_snapshot["rapg"],
        "run_diff_per_game_diff": (
            home_snapshot["run_diff_per_game"] - away_snapshot["run_diff_per_game"]
        ),
        "recent_win_pct_diff": (
            home_snapshot["recent_win_pct"] - away_snapshot["recent_win_pct"]
        ),
        "recent_rpg_diff": home_snapshot["recent_rpg"] - away_snapshot["recent_rpg"],
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


def count_lineup_starters_from_boxscore(boxscore, side):
    team_data = boxscore.get("teams", {}).get(side, {})
    players = team_data.get("players", {})

    slots = set()

    for _, player_obj in players.items():
        batting_order = player_obj.get("battingOrder")

        if batting_order is None:
            continue

        try:
            order_num = int(batting_order)
            slot = order_num // 100

            if 1 <= slot <= 9:
                slots.add(slot)
        except Exception:
            continue

    return len(slots)


@st.cache_data(ttl=60 * 5)
def get_lineup_status(game_pk):
    if not game_pk:
        return "Unknown"

    try:
        boxscore = fetch_boxscore(game_pk)
        home_count = count_lineup_starters_from_boxscore(boxscore, "home")
        away_count = count_lineup_starters_from_boxscore(boxscore, "away")

        if home_count >= 9 and away_count >= 9:
            return "Confirmed"

        if home_count >= 9 or away_count >= 9:
            return "Partial"

        return "Not Confirmed"

    except Exception:
        return "Unknown"


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

                    if key not in odds_by_team or price > odds_by_team[key]:
                        odds_by_team[key] = price

    return odds_by_team


# =========================
# Card Rendering
# =========================

def render_game_card(row, show_market_data=False):
    lineup_status = row.get("lineup_status", "Unknown")
    lineup_class = grade_lineup_status(lineup_status)

    model_tier = row.get("model_tier", "Lean")
    model_tier_class_name = tier_class(model_tier)

    edge = row.get("edge")
    price_gap_text = format_percent(edge) if edge is not None and not pd.isna(edge) else "—"

    rank = row.get("rank", None)

    if rank is not None and not pd.isna(rank):
        title_html = f'<span class="rank-badge">#{int(rank)}</span>{h(row["game"])}'
    else:
        title_html = h(row["game"])

    if show_market_data:
        box_1_label = "Fair ML"
        box_1_value = format_moneyline(row["fair_ml"])
        box_2_label = "Market ML"
        box_2_value = format_moneyline(row["market_ml"])
        box_3_label = "Price Gap"
        box_3_value = price_gap_text
        box_4_label = "Pitcher"
        box_4_value = row["probable_pitcher"]
    else:
        box_1_label = "Fair ML"
        box_1_value = format_moneyline(row["fair_ml"])
        box_2_label = "Pitcher"
        box_2_value = row["probable_pitcher"]
        box_3_label = "Lineups"
        box_3_value = lineup_status
        box_4_label = "Model Tier"
        box_4_value = model_tier

    reasons = row.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = []

    reason_html = "<br>".join([f"• {h(reason)}" for reason in reasons[:3]])

    html = "\n".join(
        [
            '<div class="game-card">',
            f'<div class="game-title">{title_html}</div>',
            f'<div class="game-subtitle">{h(row["game_time_et"])} ET • {h(row["status"])}</div>',
            '<div class="small-label">Model Winner</div>',
            f'<div class="model-side">{h(row["team"])}</div>',
            '<div class="small-label">Win Probability</div>',
            f'<div class="model-prob">{h(format_percent(row["model_prob"]))}</div>',
            '<div class="mini-grid">',
            '<div class="mini-box">',
            f'<div class="small-label">{h(box_1_label)}</div>',
            f'<div class="mini-value">{h(box_1_value)}</div>',
            '</div>',
            '<div class="mini-box">',
            f'<div class="small-label">{h(box_2_label)}</div>',
            f'<div class="mini-value" style="font-size: 14px;">{h(box_2_value)}</div>',
            '</div>',
            '<div class="mini-box">',
            f'<div class="small-label">{h(box_3_label)}</div>',
            f'<div class="mini-value">{h(box_3_value)}</div>',
            '</div>',
            '<div class="mini-box">',
            f'<div class="small-label">{h(box_4_label)}</div>',
            f'<div class="mini-value">{h(box_4_value)}</div>',
            '</div>',
            '</div>',
            '<div class="why-box">',
            '<div class="small-label">Why this side?</div>',
            f'<div class="why-line">{reason_html}</div>',
            '</div>',
            '<div class="status-row">',
            f'<span class="pill {lineup_class}">Lineups: {h(lineup_status)}</span>',
            f'<span class="pill {model_tier_class_name}">Model Tier: {h(model_tier)}</span>',
            f'<span class="pill pill-info">{h(row["side"])}</span>',
            '</div>',
            '</div>',
        ]
    )

    st.markdown(html, unsafe_allow_html=True)


def render_card_grid(df, columns=3, show_market_data=False):
    if df.empty:
        st.info("No games in this section.")
        return

    cols = st.columns(columns)

    for idx, (_, row) in enumerate(df.iterrows()):
        with cols[idx % columns]:
            render_game_card(row, show_market_data=show_market_data)


# =========================
# App
# =========================

last_updated_et = datetime.now(ZoneInfo("America/New_York")).strftime("%I:%M %p ET")

st.title("⚾ Moneyline Winners v1.0.2")
st.caption(
    f"Production Model v1 — validated team-only logistic model. "
    f"Last updated: {last_updated_et}."
)

model, model_features = load_model_artifact()

today_et = datetime.now(ZoneInfo("America/New_York")).date()

with st.sidebar:
    st.header("Controls")

    selected_date = st.date_input(
        "Game Date",
        value=today_et,
    )

    show_market_data = st.checkbox(
        "Show market / price data",
        value=False,
        help="Optional overlay. Model winner selection is still based only on win probability.",
    )

    default_odds_key = os.getenv("ODDS_API_KEY", "") or get_secret("ODDS_API_KEY")

    odds_api_key = ""

    if show_market_data:
        odds_api_key = st.text_input(
            "Odds API Key",
            value=default_odds_key,
            type="password",
            help="Optional. Used only for market price display.",
        )

    hide_final_games = st.checkbox(
        "Hide final games on game day",
        value=True,
    )

    show_live_games = st.checkbox(
        "Show live/started games",
        value=False,
    )

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.write("### Model")
    st.write("Production Feature Set:")
    st.code(", ".join(model_features), language="text")

    st.write("### Last Updated")
    st.write(last_updated_et)

    st.write("### Notes")
    st.write(
        "v1.0.2 changes the product interface only. "
        "The prediction model remains Production v1 team-only. "
        "Model winner selection is based on win probability, not odds."
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
    odds_by_team = fetch_moneyline_odds(odds_api_key) if show_market_data else {}
except Exception as e:
    odds_by_team = {}
    st.warning(f"Odds download failed. Showing model-only board. Error: {e}")


if not today_games:
    st.warning("No regular-season MLB games found for the selected date.")
    st.stop()


visible_games = []

for game in today_games:
    if selected_date == today_et and hide_final_games and is_final_game(game):
        continue

    if selected_date == today_et and not show_live_games:
        if not is_pregame(game):
            continue

    visible_games.append(game)


if not visible_games:
    st.warning("No available pregame MLB games found for the selected date.")
    st.stop()


game_rows = []
all_side_rows = []

for game in visible_games:
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

    game_status = get_game_status(game)
    game_time_et = get_game_time_et(game)
    lineup_status = get_lineup_status(game.get("gamePk"))

    home_reasons = build_reasons(features, "Home")
    away_reasons = build_reasons(features, "Away")

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
            "status": game_status,
            "game_time_et": game_time_et,
            "lineup_status": lineup_status,
            "is_pregame": is_pregame(game),
            "model_tier": grade_model_tier(home_prob),
            "reasons": home_reasons,
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
            "status": game_status,
            "game_time_et": game_time_et,
            "lineup_status": lineup_status,
            "is_pregame": is_pregame(game),
            "model_tier": grade_model_tier(away_prob),
            "reasons": away_reasons,
        },
    ]

    all_side_rows.extend(candidates)

    # Winner-picking rule:
    # choose the side with the highest model win probability.
    # Odds do not drive the model side.
    best = max(candidates, key=lambda c: c["model_prob"])

    game_rows.append(best)


board = pd.DataFrame(game_rows)
all_sides = pd.DataFrame(all_side_rows)

board["lineup_sort"] = board["lineup_status"].apply(lineup_sort_value)

pregame_board = board[board["is_pregame"] == True].copy()

best_available = pregame_board.sort_values(
    ["model_prob", "lineup_sort"],
    ascending=[False, False],
).head(6).copy()

best_available["rank"] = range(1, len(best_available) + 1)

pregame_board = pregame_board.sort_values(
    ["model_prob", "lineup_sort"],
    ascending=[False, False],
).copy()

pregame_board["rank"] = None


# =========================
# KPI Cards
# =========================

total_games = len(today_games)
available_games = len(visible_games)
confirmed_lineups = int((board["lineup_status"] == "Confirmed").sum())
high_confidence = int((board["model_prob"] >= 0.56).sum())
top_model_prob = float(board["model_prob"].max())

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">Games On Slate</div>
            <div class="big-number">{total_games}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">Available Games</div>
            <div class="big-number">{available_games}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">Confirmed Lineups</div>
            <div class="big-number">{confirmed_lineups}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">B+ Model Sides</div>
            <div class="big-number">{high_confidence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">Top Model Prob</div>
            <div class="big-number">{top_model_prob:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Card Sections
# =========================

st.subheader("Best Available Winners")
st.caption("Pregame games only. Sorted by model win probability. Odds do not drive the prediction.")
render_card_grid(best_available, columns=3, show_market_data=show_market_data)

st.divider()

st.subheader("Full Pregame Slate")
render_card_grid(pregame_board, columns=3, show_market_data=show_market_data)


# =========================
# Optional Tables
# =========================

with st.expander("Show table view"):
    display_board = board.copy()

    display_board["Model Prob"] = display_board["model_prob"].apply(format_percent)
    display_board["Fair ML"] = display_board["fair_ml"].apply(format_moneyline)
    display_board["Market ML"] = display_board["market_ml"].apply(format_moneyline)
    display_board["Market Implied"] = display_board["market_prob"].apply(format_percent)
    display_board["Price Gap"] = display_board["edge"].apply(format_percent)
    display_board["Why"] = display_board["reasons"].apply(
        lambda xs: " | ".join(xs) if isinstance(xs, list) else ""
    )

    base_cols = [
        "model_tier",
        "game",
        "team",
        "side",
        "opponent",
        "Model Prob",
        "Fair ML",
        "lineup_status",
        "probable_pitcher",
        "Why",
        "status",
        "game_time_et",
    ]

    market_cols = ["Market ML", "Market Implied", "Price Gap"]

    if show_market_data:
        final_cols = base_cols[:7] + market_cols + base_cols[7:]
    else:
        final_cols = base_cols

    display_board = display_board[final_cols].rename(
        columns={
            "model_tier": "Model Tier",
            "game": "Game",
            "team": "Model Winner",
            "side": "Home/Away",
            "opponent": "Opponent",
            "lineup_status": "Lineups",
            "probable_pitcher": "Probable Pitcher",
            "status": "Status",
            "game_time_et": "Game Time ET",
        }
    )

    st.dataframe(
        display_board,
        use_container_width=True,
        hide_index=True,
    )


with st.expander("Show both sides for every visible game"):
    side_display = all_sides.copy()
    side_display["Model Prob"] = side_display["model_prob"].apply(format_percent)
    side_display["Fair ML"] = side_display["fair_ml"].apply(format_moneyline)
    side_display["Market ML"] = side_display["market_ml"].apply(format_moneyline)
    side_display["Market Implied"] = side_display["market_prob"].apply(format_percent)
    side_display["Price Gap"] = side_display["edge"].apply(format_percent)
    side_display["Why"] = side_display["reasons"].apply(
        lambda xs: " | ".join(xs) if isinstance(xs, list) else ""
    )

    base_cols = [
        "model_tier",
        "game",
        "team",
        "side",
        "opponent",
        "Model Prob",
        "Fair ML",
        "lineup_status",
        "probable_pitcher",
        "Why",
        "status",
        "game_time_et",
    ]

    market_cols = ["Market ML", "Market Implied", "Price Gap"]

    if show_market_data:
        final_cols = base_cols[:7] + market_cols + base_cols[7:]
    else:
        final_cols = base_cols

    side_display = side_display[final_cols].rename(
        columns={
            "model_tier": "Model Tier",
            "game": "Game",
            "team": "Team",
            "side": "Home/Away",
            "opponent": "Opponent",
            "lineup_status": "Lineups",
            "probable_pitcher": "Probable Pitcher",
            "status": "Status",
            "game_time_et": "Game Time ET",
        }
    )

    st.dataframe(
        side_display,
        use_container_width=True,
        hide_index=True,
    )


st.divider()

st.caption(
    "Important: v1.0.2 changes the interface only. "
    "The prediction model remains the validated team-only Production v1 model. "
    "Market odds are optional display data and are not used to create the prediction."
)