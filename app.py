import os
import re
import html as html_lib
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests
import streamlit as st


# =========================
# App Config
# =========================

st.set_page_config(
    page_title="Moneyline Winners v1.2",
    page_icon="⚾",
    layout="wide",
)

MLB_BASE = "https://statsapi.mlb.com/api/v1"
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"

MODEL_PATH = Path("model_artifacts/mlb_candidate_v1_1_logistic_model.joblib")
TOTALS_MODEL_PATH = Path("model_artifacts/mlb_totals_projection_v0_1.joblib")

FINAL_STATES = {"Final", "Game Over", "Completed Early"}

PREGAME_STATES = {
    "Scheduled",
    "Pre-Game",
    "Warmup",
    "Delayed Start",
    "Delayed",
    "Postponed",
}

DEFAULT_FEATURES = [
    "win_pct_diff",
    "rpg_diff",
    "rapg_diff",
    "run_diff_per_game_diff",
    "recent_win_pct_diff",
    "recent_rpg_diff",
    "recent_rapg_diff",
    "home_field",
    "vs_hand_games_scaled_diff",
    "env_temp",
    "opp_adj_offense_diff",
]

CANDIDATE_ADDED_FEATURES = [
    "vs_hand_games_scaled_diff",
    "env_temp",
    "opp_adj_offense_diff",
]


MANUAL_VENUE_COORDS = {
    "yankee stadium": (40.8296, -73.9262),
    "fenway park": (42.3467, -71.0972),
    "oriole park at camden yards": (39.2840, -76.6217),
    "rogers centre": (43.6414, -79.3894),
    "comerica park": (42.3390, -83.0485),
    "progressive field": (41.4962, -81.6852),
    "target field": (44.9817, -93.2776),
    "kauffman stadium": (39.0517, -94.4803),
    "rate field": (41.8300, -87.6339),
    "guaranteed rate field": (41.8300, -87.6339),
    "pnc park": (40.4469, -80.0057),
    "great american ball park": (39.0978, -84.5066),
    "citi field": (40.7571, -73.8458),
    "wrigley field": (41.9484, -87.6553),
    "american family field": (43.0280, -87.9712),
    "busch stadium": (38.6226, -90.1928),
    "coors field": (39.7561, -104.9942),
    "chase field": (33.4455, -112.0667),
    "tropicana field": (27.7682, -82.6534),
    "loandepot park": (25.7781, -80.2197),
    "angel stadium": (33.8003, -117.8827),
    "petco park": (32.7073, -117.1573),
    "dodger stadium": (34.0739, -118.2400),
    "oracle park": (37.7786, -122.3893),
    "truist park": (33.8908, -84.4678),
    "tmobile park": (47.5914, -122.3325),
    "globe life field": (32.7473, -97.0842),
    "minute maid park": (29.7573, -95.3555),
    "daikin park": (29.7573, -95.3555),
    "oakland coliseum": (37.7516, -122.2005),
    "sutter health park": (38.5804, -121.5139),
    "citizens bank park": (39.9061, -75.1665),
    "nationals park": (38.8730, -77.0074),
}


# =========================
# Session State
# =========================

if "tracked_games" not in st.session_state:
    st.session_state["tracked_games"] = []

if "tracked_game_meta" not in st.session_state:
    st.session_state["tracked_game_meta"] = {}


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

        .track-box {
            background: rgba(143,211,255,0.08);
            border: 1px solid rgba(143,211,255,0.14);
            border-radius: 14px;
            padding: 8px 10px 4px 10px;
            margin-bottom: 7px;
        }

        .tracker-card {
            background: linear-gradient(135deg, #0d1b2e, #102944);
            border: 1px solid #1f456e;
            border-radius: 20px;
            padding: 16px 16px 14px 16px;
            margin-bottom: 16px;
            box-shadow: 0 0 18px rgba(0,0,0,0.28);
            min-height: 230px;
        }

        .tracker-head {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 10px;
            align-items: flex-start;
        }

        .tracker-game {
            color: #eaf2ff;
            font-size: 16px;
            font-weight: 900;
            line-height: 1.15;
        }

        .tracker-status {
            color: #8fd3ff;
            font-size: 12px;
            font-weight: 900;
            text-align: right;
            text-transform: uppercase;
            letter-spacing: .06em;
            min-width: 90px;
        }

        .tracker-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }

        .tracker-team {
            color: #eaf2ff;
            font-size: 20px;
            font-weight: 900;
        }

        .tracker-score {
            color: #ffffff;
            font-size: 28px;
            font-weight: 950;
        }

        .tracker-meta {
            margin-top: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(143,211,255,0.16);
            border-radius: 14px;
            padding: 10px 12px;
        }

        .tracker-progress {
            color: #ffffff;
            font-size: 16px;
            font-weight: 900;
        }

        .tracker-outs {
            color: #b8c7d9;
            font-size: 13px;
            font-weight: 700;
            margin-top: 2px;
        }

        .tracker-model-line {
            margin-top: 12px;
            display: flex;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
            color: #dceaff;
            font-size: 13px;
            font-weight: 800;
        }

        .tracker-empty {
            background: rgba(143,211,255,0.08);
            border: 1px dashed rgba(143,211,255,0.30);
            border-radius: 18px;
            padding: 22px;
            color: #dceaff;
            font-weight: 750;
        }

        .totals-card {
            background: linear-gradient(135deg, #101b2e, #1a2842);
            border: 1px solid #315a86;
            border-radius: 20px;
            padding: 18px;
            margin-bottom: 16px;
            box-shadow: 0 0 18px rgba(0,0,0,0.25);
            min-height: 290px;
        }

        .totals-main {
            font-size: 34px;
            font-weight: 950;
            color: #FFE27A;
            line-height: 1.0;
            margin-top: 4px;
            margin-bottom: 8px;
        }

        .totals-signal {
            font-size: 20px;
            font-weight: 900;
            color: #8fd3ff;
            margin-top: 5px;
            margin-bottom: 10px;
        }

        .preview-warning {
            background: rgba(255,226,122,0.08);
            border: 1px solid rgba(255,226,122,0.18);
            border-radius: 14px;
            padding: 12px 14px;
            color: #FFE27A;
            font-weight: 800;
            margin-bottom: 14px;
        }

        div[data-testid="stDataFrame"] {
            background-color: #0d1b2e;
        }

        div[data-testid="stButton"] > button,
        .stButton > button {
            background: linear-gradient(135deg, #0d1b2e, #102944) !important;
            color: #eaf2ff !important;
            border: 1px solid #1f456e !important;
            border-radius: 14px !important;
            font-weight: 900 !important;
            box-shadow: 0 0 12px rgba(0,0,0,0.18) !important;
        }

        div[data-testid="stButton"] > button:hover,
        .stButton > button:hover {
            background: linear-gradient(135deg, #12345a, #15517f) !important;
            color: #ffffff !important;
            border: 1px solid #8fd3ff !important;
        }

        div[data-testid="stButton"] > button:focus,
        .stButton > button:focus {
            color: #ffffff !important;
            border: 1px solid #8fd3ff !important;
            box-shadow: 0 0 0 2px rgba(143,211,255,0.20) !important;
        }

        div[data-testid="stButton"] > button:disabled,
        .stButton > button:disabled {
            background: rgba(255,255,255,0.08) !important;
            color: #8ea1b8 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# General Helpers
# =========================

def h(value):
    return html_lib.escape(str(value))


def normalize_key(value):
    value = str(value or "").lower().strip()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9 ]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_team_name(name):
    if not name:
        return ""

    name = name.lower()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def safe_int(x, default=0):
    try:
        if x is None or pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def safe_float(x, default=None):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def normalize_hand(value):
    if value is None or pd.isna(value):
        return "U"

    value = str(value).upper().strip()

    if value in {"L", "LEFT", "LEFTY"}:
        return "L"

    if value in {"R", "RIGHT", "RIGHTY"}:
        return "R"

    if value in {"S", "SWITCH"}:
        return "S"

    return "U"


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


def get_game_utc_datetime(game):
    game_datetime = game.get("gameDate")

    if not game_datetime:
        return None

    try:
        return datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
    except Exception:
        return None


def get_game_time_et(game):
    dt = get_game_utc_datetime(game)

    if dt is None:
        return "TBD"

    et = dt.astimezone(ZoneInfo("America/New_York"))

    try:
        return et.strftime("%-I:%M %p")
    except Exception:
        return et.strftime("%I:%M %p").lstrip("0")


def get_team_id(game, side):
    return safe_int(
        game.get("teams", {}).get(side, {}).get("team", {}).get("id")
    )


def get_team_name(game, side):
    return (
        game.get("teams", {})
        .get(side, {})
        .get("team", {})
        .get("name", side.title())
    )


def get_team_abbr(game, side):
    team = game.get("teams", {}).get(side, {}).get("team", {})
    return team.get("abbreviation") or team.get("teamName") or team.get("name", side.title())


def get_score(game, side):
    return safe_int(game.get("teams", {}).get(side, {}).get("score"), default=None)


def get_probable_pitcher_id(game, side):
    pitcher = game.get("teams", {}).get(side, {}).get("probablePitcher")

    if not pitcher:
        return 0

    return safe_int(pitcher.get("id"))


def get_probable_pitcher_name(game, side):
    pitcher = game.get("teams", {}).get(side, {}).get("probablePitcher")

    if not pitcher:
        return "TBD"

    return pitcher.get("fullName", "TBD")


def is_final_game(game):
    status = get_game_status(game)
    home_score = get_score(game, "home")
    away_score = get_score(game, "away")

    return status in FINAL_STATES and home_score is not None and away_score is not None


def is_pregame(game):
    status = get_game_status(game)
    return status in PREGAME_STATES or "Scheduled" in status or "Pre-Game" in status


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


# =========================
# Tracking Helpers
# =========================

def get_game_id_from_row(row):
    if "game_pk" in row and pd.notna(row["game_pk"]):
        return str(safe_int(row["game_pk"]))
    if "Game_PK" in row and pd.notna(row["Game_PK"]):
        return str(safe_int(row["Game_PK"]))
    return str(row.get("game", ""))


def get_game_id_from_game(game):
    return str(safe_int(game.get("gamePk")))


def is_game_tracked(game_id):
    game_id = str(game_id)
    return game_id in set(str(x) for x in st.session_state.get("tracked_games", []))


def update_tracked_games(game_id, checked, row=None):
    game_id = str(game_id)
    tracked = set(str(x) for x in st.session_state.get("tracked_games", []))
    meta = dict(st.session_state.get("tracked_game_meta", {}))

    if checked:
        tracked.add(game_id)

        if row is not None:
            meta[game_id] = {
                "game": str(row.get("game", "")),
                "team": str(row.get("team", "—")),
                "team_full": str(row.get("team_full", row.get("team", "—"))),
                "side": str(row.get("side", "—")),
                "model_prob": float(row.get("model_prob", 0.0)) if pd.notna(row.get("model_prob", None)) else None,
                "fair_ml": row.get("fair_ml", None),
                "probable_pitcher": str(row.get("probable_pitcher", "TBD")),
                "tracked_at_et": datetime.now(ZoneInfo("America/New_York")).strftime("%I:%M %p ET"),
            }
    else:
        tracked.discard(game_id)
        meta.pop(game_id, None)

    st.session_state["tracked_games"] = sorted(tracked)
    st.session_state["tracked_game_meta"] = meta


def clear_tracked_games():
    st.session_state["tracked_games"] = []
    st.session_state["tracked_game_meta"] = {}


def get_tracked_meta(game_id):
    return st.session_state.get("tracked_game_meta", {}).get(str(game_id), {})


# =========================
# Model Loading
# =========================

@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        st.error(
            f"Candidate v1.1 model file not found at: {MODEL_PATH}\n\n"
            "Make sure mlb_candidate_v1_1_logistic_model.joblib is inside model_artifacts/"
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

        model_version = artifact.get("model_version", "candidate_v1_1")

        if model is None:
            st.error("Loaded model artifact is a dictionary, but no model was found inside it.")
            st.stop()

        if features is None:
            features = infer_model_features(model)

        return model, list(features), model_version, artifact

    model = artifact
    features = infer_model_features(model)

    return model, features, "candidate_v1_1", {}



@st.cache_resource
def load_totals_artifact():
    if not TOTALS_MODEL_PATH.exists():
        return None, [], "totals_missing", {}

    artifact = joblib.load(TOTALS_MODEL_PATH)

    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("pipeline") or artifact.get("estimator")
        features = artifact.get("features") or artifact.get("feature_names") or []
        model_version = artifact.get("model_version", "totals_projection_v0_1")

        if model is None:
            return None, [], "totals_invalid", artifact

        return model, list(features), model_version, artifact

    model = artifact
    features = infer_model_features(model)
    return model, features, "totals_projection_v0_1", {}


def infer_model_features(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return DEFAULT_FEATURES.copy()


# =========================
# MLB API
# =========================

@st.cache_data(ttl=60 * 30)
def fetch_schedule_range(start_day, end_day):
    if pd.to_datetime(start_day).date() > pd.to_datetime(end_day).date():
        return {"dates": []}

    params = {
        "sportId": 1,
        "startDate": str(start_day),
        "endDate": str(end_day),
        "gameTypes": "R",
        "hydrate": "team,venue,linescore,probablePitcher",
    }

    r = requests.get(f"{MLB_BASE}/schedule", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30)
def fetch_scoreboard_games(slate_date):
    params = {
        "sportId": 1,
        "startDate": str(slate_date),
        "endDate": str(slate_date),
        "gameTypes": "R",
        "hydrate": "team,venue,linescore,probablePitcher",
    }

    r = requests.get(f"{MLB_BASE}/schedule", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return flatten_games(data)


def flatten_games(schedule_json):
    games = []

    for date_block in schedule_json.get("dates", []):
        for game in date_block.get("games", []):
            if game.get("gameType") != "R":
                continue
            games.append(game)

    return games


@st.cache_data(ttl=60 * 60)
def fetch_people(player_ids_tuple):
    player_ids = sorted(set(safe_int(pid) for pid in player_ids_tuple if safe_int(pid) > 0))

    if not player_ids:
        return {}

    people = {}
    batch_size = 100

    for i in range(0, len(player_ids), batch_size):
        batch = player_ids[i:i + batch_size]

        params = {
            "personIds": ",".join(str(pid) for pid in batch),
        }

        r = requests.get(f"{MLB_BASE}/people", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        for person in data.get("people", []):
            player_id = safe_int(person.get("id"))

            people[player_id] = {
                "full_name": person.get("fullName"),
                "pitch_hand": normalize_hand(person.get("pitchHand", {}).get("code")),
                "bat_side": normalize_hand(person.get("batSide", {}).get("code")),
            }

    return people


@st.cache_data(ttl=60 * 5)
def fetch_boxscore(game_pk):
    r = requests.get(f"{MLB_BASE}/game/{game_pk}/boxscore", timeout=20)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60 * 15)
def fetch_game_feed(game_pk):
    urls = [
        f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
        f"https://statsapi.mlb.com/api/v1/game/{game_pk}/feed/live",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            continue

    return None


# =========================
# Lineup Status
# =========================

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
# Team State Features
# =========================

def create_team_state():
    return {
        "games": 0,
        "wins": 0,
        "runs_for": 0,
        "runs_against": 0,
        "recent": deque(maxlen=10),
        "opponents": [],
        "recent_opponents": deque(maxlen=10),
    }


def update_team_state(state, runs_for, runs_against, opponent_id):
    runs_for = safe_int(runs_for)
    runs_against = safe_int(runs_against)

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
            "run_diff": runs_for - runs_against,
        }
    )

    state["opponents"].append(opponent_id)
    state["recent_opponents"].append(opponent_id)


def snapshot_team_state(state):
    games = state["games"]

    if games <= 0:
        return {
            "win_pct": 0.500,
            "rpg": 4.50,
            "rapg": 4.50,
            "run_diff_per_game": 0.00,
            "run_diff_pg": 0.00,
            "recent_win_pct": 0.500,
            "recent_rpg": 4.50,
            "recent_rapg": 4.50,
            "recent_run_diff_pg": 0.00,
            "games_scaled": 0.00,
        }

    win_pct = state["wins"] / games
    rpg = state["runs_for"] / games
    rapg = state["runs_against"] / games
    run_diff_pg = rpg - rapg

    recent = list(state["recent"])

    if recent:
        recent_win_pct = sum(g["win"] for g in recent) / len(recent)
        recent_rpg = sum(g["runs_for"] for g in recent) / len(recent)
        recent_rapg = sum(g["runs_against"] for g in recent) / len(recent)
        recent_run_diff_pg = sum(g["run_diff"] for g in recent) / len(recent)
    else:
        recent_win_pct = win_pct
        recent_rpg = rpg
        recent_rapg = rapg
        recent_run_diff_pg = run_diff_pg

    return {
        "win_pct": win_pct,
        "rpg": rpg,
        "rapg": rapg,
        "run_diff_per_game": run_diff_pg,
        "run_diff_pg": run_diff_pg,
        "recent_win_pct": recent_win_pct,
        "recent_rpg": recent_rpg,
        "recent_rapg": recent_rapg,
        "recent_run_diff_pg": recent_run_diff_pg,
        "games_scaled": min(games / 50.0, 1.0),
    }


def build_team_states_from_history(history_games):
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
        home_team_id = get_team_id(game, "home")
        away_team_id = get_team_id(game, "away")

        home_score = get_score(game, "home")
        away_score = get_score(game, "away")

        if home_score is None or away_score is None:
            continue

        update_team_state(
            team_states[home_team_id],
            home_score,
            away_score,
            opponent_id=away_team_id,
        )

        update_team_state(
            team_states[away_team_id],
            away_score,
            home_score,
            opponent_id=home_team_id,
        )

    return team_states


def schedule_strength_snapshot(team_states, opponent_ids):
    opponent_ids = list(opponent_ids)

    if not opponent_ids:
        return {
            "opp_win_pct": 0.500,
            "opp_rpg": 4.50,
            "opp_rapg": 4.50,
            "opp_run_diff_pg": 0.00,
            "opponents_faced_scaled": 0.00,
        }

    snaps = [snapshot_team_state(team_states[opp_id]) for opp_id in opponent_ids]

    return {
        "opp_win_pct": sum(s["win_pct"] for s in snaps) / len(snaps),
        "opp_rpg": sum(s["rpg"] for s in snaps) / len(snaps),
        "opp_rapg": sum(s["rapg"] for s in snaps) / len(snaps),
        "opp_run_diff_pg": sum(s["run_diff_pg"] for s in snaps) / len(snaps),
        "opponents_faced_scaled": min(len(opponent_ids) / 50.0, 1.0),
    }


def compute_opp_adj_offense_diff(team_states, home_team_id, away_team_id):
    home = snapshot_team_state(team_states[home_team_id])
    away = snapshot_team_state(team_states[away_team_id])

    home_sos = schedule_strength_snapshot(
        team_states,
        team_states[home_team_id]["opponents"],
    )

    away_sos = schedule_strength_snapshot(
        team_states,
        team_states[away_team_id]["opponents"],
    )

    home_adj_offense = home["rpg"] - home_sos["opp_rapg"]
    away_adj_offense = away["rpg"] - away_sos["opp_rapg"]

    return home_adj_offense - away_adj_offense


def build_base_team_features_for_game(game, team_states):
    home_team_id = get_team_id(game, "home")
    away_team_id = get_team_id(game, "away")

    home_snapshot = snapshot_team_state(team_states[home_team_id])
    away_snapshot = snapshot_team_state(team_states[away_team_id])

    return {
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
            away_snapshot["recent_rapg"] - home_snapshot["recent_rapg"]
        ),
        "home_field": 1.0,
    }



# =========================
# Totals Projection Preview Helpers
# =========================

def build_totals_feature_dict(game, team_states, base_features, hand_features):
    home_team_id = get_team_id(game, "home")
    away_team_id = get_team_id(game, "away")

    home_snapshot = snapshot_team_state(team_states[home_team_id])
    away_snapshot = snapshot_team_state(team_states[away_team_id])

    totals_features = dict(base_features)
    totals_features.update(hand_features)

    totals_features.update(
        {
            "home_win_pct": home_snapshot["win_pct"],
            "away_win_pct": away_snapshot["win_pct"],
            "home_rpg": home_snapshot["rpg"],
            "away_rpg": away_snapshot["rpg"],
            "home_rapg": home_snapshot["rapg"],
            "away_rapg": away_snapshot["rapg"],
            "home_run_diff_per_game": home_snapshot["run_diff_per_game"],
            "away_run_diff_per_game": away_snapshot["run_diff_per_game"],
            "home_run_diff_pg": home_snapshot["run_diff_pg"],
            "away_run_diff_pg": away_snapshot["run_diff_pg"],
            "home_recent_win_pct": home_snapshot["recent_win_pct"],
            "away_recent_win_pct": away_snapshot["recent_win_pct"],
            "home_recent_rpg": home_snapshot["recent_rpg"],
            "away_recent_rpg": away_snapshot["recent_rpg"],
            "home_recent_rapg": home_snapshot["recent_rapg"],
            "away_recent_rapg": away_snapshot["recent_rapg"],
            "home_recent_run_diff_per_game": home_snapshot["recent_run_diff_pg"],
            "away_recent_run_diff_per_game": away_snapshot["recent_run_diff_pg"],
            "home_recent_run_diff_pg": home_snapshot["recent_run_diff_pg"],
            "away_recent_run_diff_pg": away_snapshot["recent_run_diff_pg"],
        }
    )

    derived_map = {
        "win_pct_diff": "abs_win_pct_diff",
        "rpg_diff": "abs_rpg_diff",
        "rapg_diff": "abs_rapg_diff",
        "run_diff_per_game_diff": "abs_run_diff_per_game_diff",
        "recent_win_pct_diff": "abs_recent_win_pct_diff",
        "recent_rpg_diff": "abs_recent_rpg_diff",
        "recent_rapg_diff": "abs_recent_rapg_diff",
        "opp_adj_offense_diff": "abs_opp_adj_offense_diff",
        "vs_hand_games_scaled_diff": "abs_vs_hand_games_scaled_diff",
    }

    for src, dest in derived_map.items():
        totals_features[dest] = abs(float(totals_features.get(src, 0.0) or 0.0))

    totals_features["pre_total_offense_rpg_sum"] = totals_features["home_rpg"] + totals_features["away_rpg"]
    totals_features["pre_total_allowed_rpg_sum"] = totals_features["home_rapg"] + totals_features["away_rapg"]
    totals_features["pre_recent_offense_rpg_sum"] = totals_features["home_recent_rpg"] + totals_features["away_recent_rpg"]
    totals_features["pre_recent_allowed_rpg_sum"] = totals_features["home_recent_rapg"] + totals_features["away_recent_rapg"]

    return totals_features


def predict_projected_total(totals_model, totals_features, feature_dict):
    if totals_model is None or not totals_features:
        return None

    X = pd.DataFrame([feature_dict])

    for f in totals_features:
        if f not in X.columns:
            X[f] = 0.0

    X = X[totals_features].astype(float)

    try:
        return float(totals_model.predict(X)[0])
    except Exception:
        return None


def round_half_up(value):
    try:
        return round(float(value) * 2) / 2
    except Exception:
        return None


def required_book_total_for_under(projected_total, min_edge=0.50, minimum_line=9.5):
    if projected_total is None or pd.isna(projected_total):
        return None

    target = float(projected_total) + float(min_edge)
    half_line = (int(target * 2 + 0.999999) / 2.0)
    return max(float(minimum_line), half_line)


def total_watch_label(projected_total):
    if projected_total is None or pd.isna(projected_total):
        return "Totals model unavailable"

    under_edge_95 = 9.5 - float(projected_total)

    if under_edge_95 >= 0.75:
        return "Strong Under 9.5+ Watch"
    if under_edge_95 >= 0.50:
        return "Under 9.5+ Watch"
    if float(projected_total) >= 9.75:
        return "High Projection Watch"
    return "Projection Only"


# =========================
# Team vs Starter Hand Feature
# =========================

def create_hand_state():
    return {
        "games": 0,
        "wins": 0,
        "runs_for": 0,
        "runs_against": 0,
        "recent": deque(maxlen=10),
    }


def create_team_hand_state():
    return {
        "L": create_hand_state(),
        "R": create_hand_state(),
    }


def update_hand_state(state, runs_for, runs_against):
    runs_for = safe_int(runs_for)
    runs_against = safe_int(runs_against)

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
            "run_diff": runs_for - runs_against,
        }
    )


def snapshot_hand_state(state):
    games = state["games"]

    return {
        "games": games,
        "games_scaled": min(games / 50.0, 1.0) if games > 0 else 0.0,
    }


def build_team_hand_states_from_history(history_games, pitcher_hands):
    team_hand_states = defaultdict(create_team_hand_state)

    final_games = [g for g in history_games if is_final_game(g)]

    final_games = sorted(
        final_games,
        key=lambda g: (
            g.get("officialDate", ""),
            g.get("gamePk", 0),
        ),
    )

    for game in final_games:
        home_team_id = get_team_id(game, "home")
        away_team_id = get_team_id(game, "away")

        home_score = get_score(game, "home")
        away_score = get_score(game, "away")

        if home_score is None or away_score is None:
            continue

        home_starter_id = get_probable_pitcher_id(game, "home")
        away_starter_id = get_probable_pitcher_id(game, "away")

        home_starter_hand = pitcher_hands.get(home_starter_id, "U")
        away_starter_hand = pitcher_hands.get(away_starter_id, "U")

        # Home offense faced away starter hand.
        if away_starter_hand in {"L", "R"}:
            update_hand_state(
                team_hand_states[home_team_id][away_starter_hand],
                home_score,
                away_score,
            )

        # Away offense faced home starter hand.
        if home_starter_hand in {"L", "R"}:
            update_hand_state(
                team_hand_states[away_team_id][home_starter_hand],
                away_score,
                home_score,
            )

    return team_hand_states


def compute_vs_hand_games_scaled_diff(
    team_hand_states,
    home_team_id,
    away_team_id,
    home_starter_hand,
    away_starter_hand,
):
    home_starter_hand = normalize_hand(home_starter_hand)
    away_starter_hand = normalize_hand(away_starter_hand)

    if away_starter_hand in {"L", "R"}:
        home_state = team_hand_states[home_team_id][away_starter_hand]
    else:
        home_state = create_hand_state()

    if home_starter_hand in {"L", "R"}:
        away_state = team_hand_states[away_team_id][home_starter_hand]
    else:
        away_state = create_hand_state()

    home_snapshot = snapshot_hand_state(home_state)
    away_snapshot = snapshot_hand_state(away_state)

    return {
        "home_vs_hand_games_scaled": home_snapshot["games_scaled"],
        "away_vs_hand_games_scaled": away_snapshot["games_scaled"],
        "vs_hand_games_scaled_diff": (
            home_snapshot["games_scaled"] - away_snapshot["games_scaled"]
        ),
        "home_vs_hand_games": home_snapshot["games"],
        "away_vs_hand_games": away_snapshot["games"],
    }


# =========================
# Weather / env_temp
# =========================

def get_venue_id(game):
    return safe_int(game.get("venue", {}).get("id"))


def get_venue_name(game):
    return game.get("venue", {}).get("name", "Unknown Venue")


def extract_coords_from_venue_obj(venue):
    possible_locations = [
        venue.get("location", {}) if isinstance(venue, dict) else {},
        venue.get("venues", [{}])[0].get("location", {}) if isinstance(venue, dict) and venue.get("venues") else {},
    ]

    for loc in possible_locations:
        if not isinstance(loc, dict):
            continue

        possible_coord_objs = [
            loc.get("defaultCoordinates", {}),
            loc.get("coordinates", {}),
            loc,
        ]

        for coord_obj in possible_coord_objs:
            if not isinstance(coord_obj, dict):
                continue

            lat = coord_obj.get("latitude") or coord_obj.get("lat") or coord_obj.get("y")
            lon = coord_obj.get("longitude") or coord_obj.get("lng") or coord_obj.get("lon") or coord_obj.get("x")

            lat = safe_float(lat)
            lon = safe_float(lon)

            if lat is not None and lon is not None:
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return lat, lon, "mlb_venue_coordinates"

    return None, None, None


@st.cache_data(ttl=60 * 60 * 24)
def fetch_venue_details(venue_id):
    if not venue_id:
        return None

    urls = [
        f"{MLB_BASE}/venues/{venue_id}",
        f"{MLB_BASE}/venues?venueIds={venue_id}",
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            continue

    return None


def get_venue_coordinates(game):
    venue = game.get("venue", {})
    venue_name = get_venue_name(game)
    venue_id = get_venue_id(game)

    lat, lon, source = extract_coords_from_venue_obj(venue)

    if lat is not None and lon is not None:
        return lat, lon, source

    details = fetch_venue_details(venue_id)

    if details:
        lat, lon, source = extract_coords_from_venue_obj(details)

        if lat is not None and lon is not None:
            return lat, lon, source

    manual = MANUAL_VENUE_COORDS.get(normalize_key(venue_name))

    if manual:
        return manual[0], manual[1], "manual_venue_coordinates"

    return None, None, "missing_coordinates"


def parse_temp_from_value(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        temp = float(value)

        if -20 <= temp <= 130:
            return temp

        return None

    text = str(value)

    match = re.search(r"(-?\d{1,3})\s*(?:degrees|degree|deg|°)", text, re.I)

    if match:
        temp = float(match.group(1))

        if -20 <= temp <= 130:
            return temp

    return None


def extract_temperature_from_mlb_feed(feed):
    if not feed:
        return None, "mlb_feed_missing"

    weather = feed.get("gameData", {}).get("weather", {})

    for key in ["temp", "temperature"]:
        temp = parse_temp_from_value(weather.get(key))

        if temp is not None:
            return temp, "mlb_game_feed_weather"

    for value in weather.values():
        temp = parse_temp_from_value(value)

        if temp is not None:
            return temp, "mlb_game_feed_weather_text"

    info_items = (
        feed.get("liveData", {})
        .get("boxscore", {})
        .get("info", [])
    )

    for item in info_items:
        label = str(item.get("label", "")).lower()
        value = item.get("value", "")

        if "weather" in label:
            temp = parse_temp_from_value(value)

            if temp is not None:
                return temp, "mlb_boxscore_weather"

    return None, "mlb_temp_unavailable"


@st.cache_data(ttl=60 * 30)
def fetch_open_meteo_hourly_temp(lat, lon, game_utc_iso):
    game_utc_dt = datetime.fromisoformat(game_utc_iso)

    target_et_date = game_utc_dt.astimezone(ZoneInfo("America/New_York")).date()

    start_date = target_et_date - timedelta(days=1)
    end_date = target_et_date + timedelta(days=1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    r = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    if not times or not temps:
        return None, None, None, "open_meteo_missing_hourly"

    utc_offset_seconds = safe_int(data.get("utc_offset_seconds"), default=0)
    game_local_naive = (game_utc_dt + timedelta(seconds=utc_offset_seconds)).replace(tzinfo=None)

    best = None

    for t, temp in zip(times, temps):
        if temp is None:
            continue

        try:
            forecast_dt = datetime.fromisoformat(str(t))
        except Exception:
            continue

        diff_minutes = abs((forecast_dt - game_local_naive).total_seconds()) / 60.0

        if best is None or diff_minutes < best["diff_minutes"]:
            best = {
                "temp": float(temp),
                "forecast_hour": str(t),
                "diff_minutes": diff_minutes,
            }

    if best is None:
        return None, None, None, "open_meteo_no_matching_temp"

    return (
        best["temp"],
        best["forecast_hour"],
        best["diff_minutes"],
        "open_meteo_forecast",
    )


def get_env_temp_for_game(game):
    game_pk = safe_int(game.get("gamePk"))
    game_utc_dt = get_game_utc_datetime(game)

    feed = fetch_game_feed(game_pk)
    mlb_temp, mlb_source = extract_temperature_from_mlb_feed(feed)

    if mlb_temp is not None:
        return {
            "env_temp": mlb_temp,
            "env_temp_source": mlb_source,
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }

    lat, lon, coord_source = get_venue_coordinates(game)

    if lat is None or lon is None or game_utc_dt is None:
        return {
            "env_temp": 74.0,
            "env_temp_source": "default_missing_coordinates_or_time",
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }

    try:
        forecast_temp, forecast_hour, diff_minutes, forecast_source = (
            fetch_open_meteo_hourly_temp(
                float(lat),
                float(lon),
                game_utc_dt.isoformat(),
            )
        )

        if forecast_temp is not None:
            return {
                "env_temp": forecast_temp,
                "env_temp_source": forecast_source,
                "forecast_hour": forecast_hour,
                "forecast_hour_diff_minutes": round(diff_minutes, 1),
            }

        return {
            "env_temp": 74.0,
            "env_temp_source": f"default_after_{forecast_source}",
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }

    except Exception as e:
        return {
            "env_temp": 74.0,
            "env_temp_source": f"default_after_open_meteo_error:{type(e).__name__}",
            "forecast_hour": "",
            "forecast_hour_diff_minutes": "",
        }


# =========================
# Live Outcome Tracker
# =========================

def get_scoreboard_team(game, side):
    team = game.get("teams", {}).get(side, {})
    team_obj = team.get("team", {})
    return team_obj.get("abbreviation") or team_obj.get("teamName") or team_obj.get("name", side.title())


def get_scoreboard_score(game, side):
    score = game.get("teams", {}).get(side, {}).get("score")
    if score is None:
        return "—"
    return str(score)


def get_game_progress(game):
    status = get_game_status(game)
    linescore = game.get("linescore", {}) or {}

    if status in FINAL_STATES:
        return "Final", ""

    current_inning = linescore.get("currentInning")
    inning_state = linescore.get("inningState", "")
    outs = linescore.get("outs", None)

    if current_inning:
        inning_text = f"{inning_state} {current_inning}".strip()
        if outs is None:
            outs_text = ""
        elif int(outs) == 1:
            outs_text = "1 Out"
        else:
            outs_text = f"{int(outs)} Outs"
        return inning_text, outs_text

    if status in {"Scheduled", "Pre-Game", "Warmup"}:
        return f"{get_game_time_et(game)} ET", status

    return status, ""


def get_bases_html(game):
    linescore = game.get("linescore", {}) or {}
    offense = linescore.get("offense", {}) or {}

    on_first = offense.get("first") is not None
    on_second = offense.get("second") is not None
    on_third = offense.get("third") is not None

    def diamond(filled):
        if filled:
            return '<span style="color:#7CFFB2;font-size:1.05rem;">◆</span>'
        return '<span style="color:#64748b;font-size:1.05rem;">◇</span>'

    return (
        '<div style="line-height:1.0;text-align:center;min-width:44px;">'
        f'<div>{diamond(on_second)}</div>'
        f'<div>{diamond(on_third)} {diamond(on_first)}</div>'
        '</div>'
    )


def build_tracked_games_for_display(slate_date):
    tracked_ids = set(str(x) for x in st.session_state.get("tracked_games", []))

    if not tracked_ids:
        return []

    games = fetch_scoreboard_games(slate_date)
    tracked_games = [g for g in games if get_game_id_from_game(g) in tracked_ids]

    tracked_games = sorted(tracked_games, key=lambda g: g.get("gameDate", ""))
    return tracked_games


def render_live_tracker_card(game):
    game_id = get_game_id_from_game(game)
    meta = get_tracked_meta(game_id)

    away_abbr = get_scoreboard_team(game, "away")
    home_abbr = get_scoreboard_team(game, "home")

    away_score = get_scoreboard_score(game, "away")
    home_score = get_scoreboard_score(game, "home")

    status = get_game_status(game)
    progress_text, outs_text = get_game_progress(game)
    bases_html = get_bases_html(game)

    game_label = meta.get("game") or f"{away_abbr} @ {home_abbr}"
    pick_team = meta.get("team", "—")
    model_prob = meta.get("model_prob", None)
    prob_text = format_percent(model_prob) if model_prob is not None else "—"
    tracked_at = meta.get("tracked_at_et", "—")

    # Build as joined strings instead of an indented triple-quoted block.
    # Streamlit markdown can treat indented HTML as a code block.
    card_html = "\n".join(
        [
            '<div class="tracker-card">',
            '<div class="tracker-head">',
            f'<div class="tracker-game">{h(game_label)}</div>',
            f'<div class="tracker-status">{h(status)}</div>',
            '</div>',
            '<div class="tracker-row">',
            f'<div class="tracker-team">{h(away_abbr)}</div>',
            f'<div class="tracker-score">{h(away_score)}</div>',
            '</div>',
            '<div class="tracker-row">',
            f'<div class="tracker-team">{h(home_abbr)}</div>',
            f'<div class="tracker-score">{h(home_score)}</div>',
            '</div>',
            '<div class="tracker-meta">',
            '<div>',
            f'<div class="tracker-progress">{h(progress_text)}</div>',
            f'<div class="tracker-outs">{h(outs_text)}</div>',
            '</div>',
            f'<div>{bases_html}</div>',
            '</div>',
            '<div class="tracker-model-line">',
            f'<span><b>Model Pick:</b> {h(pick_team)}</span>',
            f'<span><b>Win Prob:</b> {h(prob_text)}</span>',
            f'<span><b>Tracked:</b> {h(tracked_at)}</span>',
            '</div>',
            '</div>',
        ]
    )

    st.markdown(card_html, unsafe_allow_html=True)


def render_tracker_grid(slate_date):
    tracked_games = build_tracked_games_for_display(slate_date)

    if not tracked_games:
        st.markdown(
            """
            <div class="tracker-empty">
                No tracked games yet. Go to the Winner Board and check <b>Track?</b> on any game you want to follow live.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    cols_per_row = 3

    for start in range(0, len(tracked_games), cols_per_row):
        cols = st.columns(cols_per_row)
        chunk = tracked_games[start:start + cols_per_row]

        for col, game in zip(cols, chunk):
            with col:
                game_id = get_game_id_from_game(game)
                keep_tracking = st.checkbox(
                    "Tracking",
                    value=True,
                    key=f"tracker_keep_{game_id}",
                    help="Uncheck to remove this game from the tracker.",
                )

                if not keep_tracking:
                    update_tracked_games(game_id, False)
                    st.rerun()

                render_live_tracker_card(game)


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
# Reasons
# =========================

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
        ("Recent run prevention edge", features.get("recent_rapg_diff", 0), 0.30, True),
        ("Opponent-adjusted offense edge", features.get("opp_adj_offense_diff", 0), 0.25, True),
    ]

    for label, value, threshold, home_positive in checks:
        if abs(value) >= threshold and favors_selected(value, home_positive=home_positive):
            reasons.append(label)

    if not reasons:
        reasons.append("Narrow model edge")
        reasons.append("No major single-driver advantage")

    return reasons[:3]


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


def render_track_control(row, section_key):
    game_id = get_game_id_from_row(row)
    currently_tracked = is_game_tracked(game_id)

    st.markdown('<div class="track-box">', unsafe_allow_html=True)

    # Important:
    # The same game can appear in both Best Available Winners and Full Pregame Slate.
    # If every unchecked duplicate checkbox is allowed to remove the game, tracking gets erased.
    # So winner-board checkboxes are ADD-ONLY.
    # Removal happens from the Live Outcome Tracker tab or sidebar Clear button.
    widget_state = "tracked" if currently_tracked else "open"

    checked = st.checkbox(
        "Track?",
        value=currently_tracked,
        key=f"track_{section_key}_{game_id}_{widget_state}",
        disabled=currently_tracked,
        help=(
            "Add this game to the Live Outcome Tracker tab. "
            "Remove tracked games from the tracker tab or sidebar."
        ),
    )

    if checked and not currently_tracked:
        update_tracked_games(game_id, True, row=row)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_game_card_with_tracker(row, section_key, show_market_data=False):
    render_track_control(row, section_key)
    render_game_card(row, show_market_data=show_market_data)


def render_card_grid(df, columns=3, show_market_data=False, section_key="board"):
    if df.empty:
        st.info("No games in this section.")
        return

    cols = st.columns(columns)

    for idx, (_, row) in enumerate(df.iterrows()):
        with cols[idx % columns]:
            render_game_card_with_tracker(
                row,
                section_key=f"{section_key}_{idx}",
                show_market_data=show_market_data,
            )




def render_totals_card(row):
    projected_total = row.get("projected_total_runs", None)
    projected_text = f"{projected_total:.1f}" if projected_total is not None and pd.notna(projected_total) else "—"

    edge_95 = row.get("under_edge_at_9_5", None)
    edge_95_text = f"+{edge_95:.1f}" if edge_95 is not None and pd.notna(edge_95) and edge_95 > 0 else f"{edge_95:.1f}" if edge_95 is not None and pd.notna(edge_95) else "—"

    needed_line = row.get("needed_book_total_for_under", None)
    needed_line_text = f"{needed_line:.1f}+" if needed_line is not None and pd.notna(needed_line) else "—"

    signal = row.get("total_signal", "Projection Only")
    signal_class = "pill-good" if "Under" in signal else "pill-info"

    card_html = "\n".join(
        [
            '<div class="totals-card">',
            f'<div class="game-title">{h(row.get("game", ""))}</div>',
            f'<div class="game-subtitle">{h(row.get("game_time_et", "TBD"))} ET • {h(row.get("status", ""))}</div>',
            '<div class="small-label">Projected Total Runs</div>',
            f'<div class="totals-main">{h(projected_text)}</div>',
            '<div class="small-label">Research Preview Signal</div>',
            f'<div class="totals-signal">{h(signal)}</div>',
            '<div class="mini-grid">',
            '<div class="mini-box">',
            '<div class="small-label">Needs Book Total</div>',
            f'<div class="mini-value">{h(needed_line_text)}</div>',
            '</div>',
            '<div class="mini-box">',
            '<div class="small-label">Under Edge @ 9.5</div>',
            f'<div class="mini-value">{h(edge_95_text)}</div>',
            '</div>',
            '<div class="mini-box">',
            '<div class="small-label">Weather</div>',
            f'<div class="mini-value">{h(row.get("env_temp_text", "—"))}</div>',
            '</div>',
            '<div class="mini-box">',
            '<div class="small-label">Status</div>',
            f'<div class="mini-value" style="font-size: 14px;">{h(row.get("lineup_status", "Unknown"))}</div>',
            '</div>',
            '</div>',
            '<div class="status-row">',
            f'<span class="pill {signal_class}">{h(signal)}</span>',
            '<span class="pill pill-warn">No real book line yet</span>',
            '</div>',
            '</div>',
        ]
    )

    st.markdown(card_html, unsafe_allow_html=True)


def render_totals_grid(df, columns=3):
    if df.empty:
        st.info("No totals projections available.")
        return

    cols = st.columns(columns)

    for idx, (_, row) in enumerate(df.iterrows()):
        with cols[idx % columns]:
            render_totals_card(row)


def render_mixed_parlay_preview(top_ml, totals_candidate):
    st.markdown(
        """
        <div class="preview-warning">
            Research Preview Only. This does not calculate real sportsbook edge yet because no actual total line is connected.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if top_ml.empty or len(top_ml) < 3:
        st.info("Need at least three ML picks at 58%+ to build the mixed preview.")
        return

    if totals_candidate.empty:
        st.info("No high-total Under watch candidate found outside the top 3 ML games.")
        return

    c1, c2 = st.columns([0.62, 0.38])

    with c1:
        st.write("### Top 3 ML Legs")
        ml_display = top_ml.copy()
        ml_display["Win Prob"] = ml_display["model_prob"].apply(format_percent)
        ml_display["Fair ML"] = ml_display["fair_ml"].apply(format_moneyline)
        ml_display = ml_display[["game", "team", "side", "Win Prob", "Fair ML"]].rename(
            columns={
                "game": "Game",
                "team": "Pick",
                "side": "Home/Away",
            }
        )
        st.dataframe(ml_display, use_container_width=True, hide_index=True)

    with c2:
        st.write("### Total Watch Leg")
        render_totals_card(totals_candidate.iloc[0])

    st.caption(
        "Current concept rule: top 3 ML picks at 58%+ plus one high-total Under watch leg. "
        "A real book total and price are still required before this can become a qualified betting edge."
    )

# =========================
# App
# =========================

last_updated_et = datetime.now(ZoneInfo("America/New_York")).strftime("%I:%M %p ET")

st.title("⚾ Moneyline Winners v1.2")
st.caption(
    f"Production Model v1.1 — team strength + weather + opponent-adjusted offense + hand-matchup sample. "
    f"UI v1.2 adds Totals Projection Preview. Last updated: {last_updated_et}."
)

model, model_features, model_version, model_artifact = load_model_artifact()
totals_model, totals_features, totals_model_version, totals_artifact = load_totals_artifact()

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

    st.write("### Live Outcome Tracker")
    st.write(f"Tracked games: **{len(st.session_state.get('tracked_games', []))}**")

    if st.button("Clear Tracked Games", use_container_width=True):
        clear_tracked_games()
        st.success("Tracked games cleared.")
        st.rerun()

    st.divider()

    st.write("### Model")
    st.write(f"Version: `{model_version}`")
    st.write("Production Feature Set:")
    st.code(", ".join(model_features), language="text")

    st.write("### v1.1 Added Features")
    st.write("- `env_temp`")
    st.write("- `opp_adj_offense_diff`")
    st.write("- `vs_hand_games_scaled_diff`")

    st.write("### Totals Preview")
    if totals_model is None:
        st.warning("Totals model artifact not found. Copy mlb_totals_projection_v0_1.joblib into model_artifacts/.")
    else:
        st.write(f"Version: `{totals_model_version}`")
        st.write(f"Features: `{len(totals_features)}`")

    st.write("### Last Updated")
    st.write(last_updated_et)


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
    st.info("Tracked games can still be viewed in the Live Outcome Tracker if you already added them earlier.")


all_pitcher_ids = []

for game in history_games + today_games:
    all_pitcher_ids.append(get_probable_pitcher_id(game, "home"))
    all_pitcher_ids.append(get_probable_pitcher_id(game, "away"))

try:
    people = fetch_people(tuple(sorted(set(all_pitcher_ids))))
except Exception as e:
    st.warning(f"Pitcher handedness download failed. Hand feature will use neutral defaults. Error: {e}")
    people = {}

pitcher_hands = {
    pid: info.get("pitch_hand", "U")
    for pid, info in people.items()
}

team_states = build_team_states_from_history(history_games)
team_hand_states = build_team_hand_states_from_history(history_games, pitcher_hands)

try:
    odds_by_team = fetch_moneyline_odds(odds_api_key) if show_market_data else {}
except Exception as e:
    odds_by_team = {}
    st.warning(f"Odds download failed. Showing model-only board. Error: {e}")


game_rows = []
all_side_rows = []
weather_sources = []
total_preview_rows = []


for game in visible_games:
    game_pk = safe_int(game.get("gamePk"))

    home_team_id = get_team_id(game, "home")
    away_team_id = get_team_id(game, "away")

    home_name = get_team_name(game, "home")
    away_name = get_team_name(game, "away")

    home_abbr = get_team_abbr(game, "home")
    away_abbr = get_team_abbr(game, "away")

    game_label = f"{away_abbr} @ {home_abbr}"

    home_starter_id = get_probable_pitcher_id(game, "home")
    away_starter_id = get_probable_pitcher_id(game, "away")

    home_starter_hand = pitcher_hands.get(home_starter_id, "U")
    away_starter_hand = pitcher_hands.get(away_starter_id, "U")

    features = build_base_team_features_for_game(game, team_states)

    env = get_env_temp_for_game(game)
    env_temp = env["env_temp"]
    env_temp_source = env["env_temp_source"]
    weather_sources.append(env_temp_source)

    hand_features = compute_vs_hand_games_scaled_diff(
        team_hand_states=team_hand_states,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_starter_hand=home_starter_hand,
        away_starter_hand=away_starter_hand,
    )

    features.update(
        {
            "env_temp": env_temp,
            "opp_adj_offense_diff": compute_opp_adj_offense_diff(
                team_states,
                home_team_id,
                away_team_id,
            ),
            **hand_features,
        }
    )

    totals_feature_dict = build_totals_feature_dict(
        game=game,
        team_states=team_states,
        base_features=features,
        hand_features=hand_features,
    )

    projected_total_runs = predict_projected_total(
        totals_model=totals_model,
        totals_features=totals_features,
        feature_dict=totals_feature_dict,
    )

    under_edge_at_9_5 = None if projected_total_runs is None else 9.5 - projected_total_runs
    under_edge_at_10_5 = None if projected_total_runs is None else 10.5 - projected_total_runs
    needed_book_total = required_book_total_for_under(projected_total_runs, min_edge=0.50, minimum_line=9.5)
    total_signal = total_watch_label(projected_total_runs)

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

    total_preview_rows.append(
        {
            "game_pk": game_pk,
            "game": game_label,
            "away_team": away_abbr,
            "home_team": home_abbr,
            "projected_total_runs": projected_total_runs,
            "needed_book_total_for_under": needed_book_total,
            "under_edge_at_9_5": under_edge_at_9_5,
            "under_edge_at_10_5": under_edge_at_10_5,
            "total_signal": total_signal,
            "status": game_status,
            "game_time_et": game_time_et,
            "lineup_status": lineup_status,
            "env_temp": env_temp,
            "env_temp_text": f"{env_temp:.1f}°" if env_temp is not None else "—",
            "env_temp_source": env_temp_source,
            "research_preview_only": True,
        }
    )

    candidates = [
        {
            "game_pk": game_pk,
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
            "probable_pitcher": get_probable_pitcher_name(game, "home"),
            "status": game_status,
            "game_time_et": game_time_et,
            "lineup_status": lineup_status,
            "is_pregame": is_pregame(game),
            "model_tier": grade_model_tier(home_prob),
            "reasons": home_reasons,
            "env_temp": env_temp,
            "env_temp_source": env_temp_source,
            "opp_adj_offense_diff": features["opp_adj_offense_diff"],
            "vs_hand_games_scaled_diff": features["vs_hand_games_scaled_diff"],
            "home_vs_hand_games": features["home_vs_hand_games"],
            "away_vs_hand_games": features["away_vs_hand_games"],
            "home_starter_hand": home_starter_hand,
            "away_starter_hand": away_starter_hand,
        },
        {
            "game_pk": game_pk,
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
            "probable_pitcher": get_probable_pitcher_name(game, "away"),
            "status": game_status,
            "game_time_et": game_time_et,
            "lineup_status": lineup_status,
            "is_pregame": is_pregame(game),
            "model_tier": grade_model_tier(away_prob),
            "reasons": away_reasons,
            "env_temp": env_temp,
            "env_temp_source": env_temp_source,
            "opp_adj_offense_diff": features["opp_adj_offense_diff"],
            "vs_hand_games_scaled_diff": features["vs_hand_games_scaled_diff"],
            "home_vs_hand_games": features["home_vs_hand_games"],
            "away_vs_hand_games": features["away_vs_hand_games"],
            "home_starter_hand": home_starter_hand,
            "away_starter_hand": away_starter_hand,
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
totals_preview = pd.DataFrame(total_preview_rows)

if not totals_preview.empty:
    totals_preview = totals_preview.sort_values(
        ["under_edge_at_9_5", "projected_total_runs"],
        ascending=[False, True],
    ).copy()

if board.empty:
    board = pd.DataFrame(
        columns=[
            "game_pk", "game", "team", "team_full", "opponent", "side",
            "model_prob", "fair_ml", "market_ml", "market_prob", "edge",
            "probable_pitcher", "status", "game_time_et", "lineup_status",
            "is_pregame", "model_tier", "reasons", "env_temp",
            "env_temp_source", "opp_adj_offense_diff",
            "vs_hand_games_scaled_diff", "home_vs_hand_games",
            "away_vs_hand_games", "home_starter_hand", "away_starter_hand",
        ]
    )

if all_sides.empty:
    all_sides = board.copy()

if totals_preview.empty:
    totals_preview = pd.DataFrame(
        columns=[
            "game_pk", "game", "away_team", "home_team", "projected_total_runs",
            "needed_book_total_for_under", "under_edge_at_9_5", "under_edge_at_10_5",
            "total_signal", "status", "game_time_et", "lineup_status", "env_temp",
            "env_temp_text", "env_temp_source", "research_preview_only",
        ]
    )

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

pregame_totals_preview = totals_preview[totals_preview["status"].apply(lambda s: s in PREGAME_STATES or "Scheduled" in str(s) or "Pre-Game" in str(s))].copy()

if not pregame_totals_preview.empty:
    under_watch = pregame_totals_preview[
        pd.to_numeric(pregame_totals_preview["under_edge_at_9_5"], errors="coerce") >= 0.50
    ].copy()
    under_watch = under_watch.sort_values("under_edge_at_9_5", ascending=False).copy()
else:
    under_watch = pd.DataFrame(columns=totals_preview.columns)


# =========================
# KPI Cards
# =========================

total_games = len(today_games)
available_games = len(visible_games)
confirmed_lineups = int((board["lineup_status"] == "Confirmed").sum()) if not board.empty else 0
high_confidence = int((board["model_prob"] >= 0.56).sum()) if not board.empty else 0
top_model_prob = float(board["model_prob"].max()) if not board.empty else 0.0
tracked_count = len(st.session_state.get("tracked_games", []))

weather_real_mask = ~board["env_temp_source"].astype(str).str.contains("default", case=False, na=False) if not board.empty else pd.Series([], dtype=bool)
real_weather_count = int(weather_real_mask.sum()) if not board.empty else 0

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
            <div class="small-label">Tracked Games</div>
            <div class="big-number">{tracked_count}</div>
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
# Main Tabs
# =========================

winner_tab, totals_tab, mixed_tab, tracker_tab = st.tabs(["Winner Board", "Totals Projection Preview", "Mixed Parlay Preview", "Live Outcome Tracker"])

with winner_tab:
    st.subheader("Best Available Winners")
    st.caption(
        "Pregame games only. Sorted by model win probability. "
        "Use Track? to add games to the Live Outcome Tracker. "
        "Odds do not drive the prediction."
    )
    render_card_grid(
        best_available,
        columns=3,
        show_market_data=show_market_data,
        section_key="best_available",
    )

    st.divider()

    st.subheader("Full Pregame Slate")
    st.caption("Use Track? on any lower-ranked game you still want to follow.")
    render_card_grid(
        pregame_board,
        columns=3,
        show_market_data=show_market_data,
        section_key="pregame_board",
    )

with totals_tab:
    st.subheader("Totals Projection Preview")
    st.caption(
        "Research preview only. This projects combined runs. It does not calculate real betting edge until an actual sportsbook total line is connected."
    )

    if totals_model is None:
        st.warning("Totals model artifact is missing. Copy mlb_totals_projection_v0_1.joblib into model_artifacts/ and refresh.")
    else:
        st.markdown(
            """
            <div class="preview-warning">
                No real book line connected yet. Under watch means: if the market total is 9.5 or higher, compare it to our projected total.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("### High-Total Under Watch")
        render_totals_grid(under_watch.head(6), columns=3)

        st.divider()
        st.write("### Full Totals Projection Slate")
        render_totals_grid(pregame_totals_preview.head(12), columns=3)

with mixed_tab:
    st.subheader("Mixed Parlay Preview")
    st.caption(
        "Concept from research: top 3 ML picks at 58%+ plus one high-total Under watch candidate. This is not a betting recommendation until real odds are connected."
    )

    top_3_ml = pregame_board[pregame_board["model_prob"] >= 0.58].copy().head(3)
    top_3_game_ids = set(str(x) for x in top_3_ml.get("game_pk", []))
    mixed_total_candidate = under_watch[~under_watch["game_pk"].astype(str).isin(top_3_game_ids)].head(1).copy()
    render_mixed_parlay_preview(top_3_ml, mixed_total_candidate)

with tracker_tab:
    st.subheader("Live Outcome Tracker")
    st.caption(
        "Only games you marked with Track? appear here. "
        "This tracker ignores the winner-board hide-live/final filters."
    )

    tracker_left, tracker_right = st.columns([0.72, 0.28])

    with tracker_left:
        st.write(f"Tracking **{len(st.session_state.get('tracked_games', []))}** games for {selected_date.isoformat()}.")

    with tracker_right:
        if st.button("🔄 Refresh Tracker", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    render_tracker_grid(selected_date)


# =========================
# Optional Tables
# =========================

with st.expander("Show v1.1 feature diagnostics"):
    diagnostics = board.copy()

    diagnostics["Model Prob"] = diagnostics["model_prob"].apply(format_percent)
    diagnostics["Env Temp"] = diagnostics["env_temp"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    diagnostics["Opp Adj Offense Diff"] = diagnostics["opp_adj_offense_diff"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    diagnostics["Hand Sample Diff"] = diagnostics["vs_hand_games_scaled_diff"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    diagnostics = diagnostics[
        [
            "game",
            "team",
            "Model Prob",
            "Env Temp",
            "env_temp_source",
            "Opp Adj Offense Diff",
            "Hand Sample Diff",
            "home_vs_hand_games",
            "away_vs_hand_games",
            "home_starter_hand",
            "away_starter_hand",
            "lineup_status",
            "status",
            "game_time_et",
        ]
    ].rename(
        columns={
            "game": "Game",
            "team": "Model Winner",
            "env_temp_source": "Weather Source",
            "home_vs_hand_games": "Home vs Hand Games",
            "away_vs_hand_games": "Away vs Hand Games",
            "home_starter_hand": "Home SP Hand",
            "away_starter_hand": "Away SP Hand",
            "lineup_status": "Lineups",
            "status": "Status",
            "game_time_et": "Game Time ET",
        }
    )

    st.dataframe(
        diagnostics,
        use_container_width=True,
        hide_index=True,
    )


with st.expander("Show totals projection table"):
    totals_table = totals_preview.copy()

    if not totals_table.empty:
        totals_table["Projected Total"] = totals_table["projected_total_runs"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        totals_table["Needs Book Total"] = totals_table["needed_book_total_for_under"].apply(lambda x: f"{x:.1f}+" if pd.notna(x) else "—")
        totals_table["Under Edge @ 9.5"] = totals_table["under_edge_at_9_5"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
        totals_table["Under Edge @ 10.5"] = totals_table["under_edge_at_10_5"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
        totals_table["Env Temp"] = totals_table["env_temp"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

        totals_table = totals_table[
            [
                "game",
                "Projected Total",
                "Needs Book Total",
                "Under Edge @ 9.5",
                "Under Edge @ 10.5",
                "total_signal",
                "Env Temp",
                "env_temp_source",
                "lineup_status",
                "status",
                "game_time_et",
            ]
        ].rename(
            columns={
                "game": "Game",
                "total_signal": "Preview Signal",
                "env_temp_source": "Weather Source",
                "lineup_status": "Lineups",
                "status": "Status",
                "game_time_et": "Game Time ET",
            }
        )

    st.dataframe(totals_table, use_container_width=True, hide_index=True)


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
    "Moneyline Winners v1.2. "
    "Model winner selection is based only on model win probability. "
    "Market odds are optional display data and are not used to create the prediction. "
    "Totals Projection Preview is research-only until real sportsbook totals are connected. Live Outcome Tracker is for monitoring selected games only."
)
