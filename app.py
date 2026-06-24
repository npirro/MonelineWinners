
import streamlit as st
import pandas as pd
import requests
from datetime import date, datetime
import math
import time

st.set_page_config(
    page_title="Moneyline Winners",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

MLB_BASE = "https://statsapi.mlb.com/api/v1"

DEFAULT_WEIGHTS = {
    "SP_Score": 40,
    "Offense_Score": 25,
    "Bullpen_Score": 15,
    "Lineup_Score": 12,
    "Situational_Score": 8
}

ENGINE_COLUMNS = [
    "Game", "Team", "Opponent", "Home", "Team_ID", "Opponent_ID", "Game_PK",
    "Starting_Pitcher", "SP_Score", "Offense_Score", "Bullpen_Score",
    "Lineup_Score", "Situational_Score", "Moneyline", "Notes"
]

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #05070d !important;
        color: #f8fafc !important;
    }

    [data-testid="stHeader"] {
        background: rgba(5, 7, 13, 0.96) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070b13 0%, #0b1020 100%) !important;
        border-right: 1px solid rgba(96,165,250,0.18);
    }

    [data-testid="stSidebar"] > div { padding-top: 1.1rem; }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #f8fafc !important;
    }

    [data-testid="stNumberInput"],
    [data-testid="stDateInput"] {
        background: linear-gradient(145deg, #0b1020, #111827);
        border: 1px solid rgba(96,165,250,0.18);
        border-radius: 18px;
        padding: 12px 12px 10px;
        box-shadow: 0 14px 30px rgba(0,0,0,.24);
        margin-bottom: 12px;
    }

    [data-testid="stNumberInput"] label p,
    [data-testid="stDateInput"] label p {
        color: #dbeafe !important;
        font-weight: 850 !important;
        font-size: .92rem !important;
    }

    [data-testid="stNumberInput"] input {
        background: #ffffff !important;
        color: #0f172a !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
    }

    .block-container {
        padding-top: 1.4rem;
        max-width: 1550px;
    }

    .hero-card {
        background: radial-gradient(circle at top left, rgba(59,130,246,.32), transparent 34%),
                    linear-gradient(135deg, #0b1020 0%, #111827 55%, #172033 100%);
        border: 1px solid rgba(96,165,250,0.22);
        border-radius: 26px;
        padding: 30px 34px;
        margin-bottom: 18px;
        box-shadow: 0 22px 55px rgba(0,0,0,.42);
    }

    .hero-title {
        font-size: 3.15rem;
        line-height: 1;
        font-weight: 950;
        color: #ffffff;
        margin: 0 0 10px 0;
        letter-spacing: -.04em;
        text-shadow: 0 3px 22px rgba(0,0,0,.35);
    }

    .hero-subtitle {
        color: #e2e8f0;
        font-size: 1.08rem;
        font-weight: 750;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #0b1020 0%, #111827 70%, #151f32 100%);
        border: 1px solid rgba(96,165,250,0.22);
        border-left: 7px solid #3b82f6;
        padding: 20px 22px;
        border-radius: 22px;
        box-shadow: 0 18px 38px rgba(0,0,0,0.34);
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span {
        color: #bfdbfe !important;
        opacity: 1 !important;
        font-weight: 950 !important;
        font-size: .88rem !important;
        text-transform: uppercase !important;
        letter-spacing: .11em !important;
        text-shadow: 0 2px 14px rgba(0,0,0,.55);
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.3rem !important;
        font-weight: 950 !important;
        text-shadow: 0 2px 18px rgba(0,0,0,.45);
    }

    .info-strip {
        background: rgba(37,99,235,.16);
        border: 1px solid rgba(96,165,250,.38);
        color: #eff6ff;
        border-radius: 18px;
        padding: 14px 17px;
        margin: 12px 0 20px 0;
        font-weight: 850;
        box-shadow: 0 14px 30px rgba(0,0,0,.18);
    }

    .warning-strip {
        background: rgba(245,158,11,.14);
        border: 1px solid rgba(245,158,11,.34);
        color: #fde68a;
        border-radius: 18px;
        padding: 14px 17px;
        margin: 12px 0 20px 0;
        font-weight: 800;
    }

    .section-title {
        font-size: 1.65rem;
        font-weight: 950;
        color: #ffffff;
        margin: 24px 0 12px 0;
        letter-spacing: -.02em;
    }

    .team-card {
        background: linear-gradient(145deg, #0b1020 0%, #111827 70%, #151f32 100%);
        border: 1px solid rgba(96,165,250,0.18);
        border-radius: 22px;
        padding: 19px 20px;
        min-height: 265px;
        box-shadow: 0 18px 38px rgba(0,0,0,0.34);
        margin-bottom: 16px;
    }

    .team-card-a { border-left: 8px solid #22c55e; }
    .team-card-b { border-left: 8px solid #3b82f6; }
    .team-card-c { border-left: 8px solid #f59e0b; }
    .team-card-pass { border-left: 8px solid #ef4444; }

    .rank-line {
        color: #bfdbfe;
        font-size: .78rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .11em;
        margin-bottom: 12px;
    }

    .team-name {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 950;
        line-height: 1;
        margin-bottom: 6px;
        text-shadow: 0 3px 18px rgba(0,0,0,.38);
    }

    .matchup {
        color: #dbeafe;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .pitcher {
        color: #cbd5e1;
        font-size: .87rem;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .score-flex {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 12px 0;
    }

    .score-label {
        color: #bfdbfe;
        font-size: .75rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .win-score {
        color: #ffffff;
        font-size: 2.35rem;
        font-weight: 950;
        line-height: 1;
    }

    .grade-pill {
        border-radius: 999px;
        padding: 8px 14px;
        font-weight: 950;
        font-size: .9rem;
    }

    .pill-a { background: rgba(34,197,94,.2); color: #86efac; border: 1px solid rgba(34,197,94,.35); }
    .pill-b { background: rgba(59,130,246,.2); color: #bfdbfe; border: 1px solid rgba(59,130,246,.35); }
    .pill-c { background: rgba(245,158,11,.2); color: #fde68a; border: 1px solid rgba(245,158,11,.35); }
    .pill-pass { background: rgba(239,68,68,.2); color: #fecaca; border: 1px solid rgba(239,68,68,.35); }

    .odds {
        color: #ffffff;
        font-weight: 950;
        margin-top: 12px;
        font-size: 1rem;
    }

    .odds-note {
        color: #cbd5e1;
        font-size: .83rem;
        margin-top: 2px;
        line-height: 1.3;
        font-weight: 650;
    }

    .footer-note {
        color: #dbeafe;
        font-size: .9rem;
        line-height: 1.38;
        margin-top: 13px;
        font-weight: 650;
    }

    .footer-note b { color: #ffffff; }

    .sidebar-card {
        background: linear-gradient(145deg, #0b1020, #111827);
        border: 1px solid rgba(96,165,250,0.20);
        border-radius: 20px;
        padding: 16px 16px;
        margin: 10px 0 18px;
        box-shadow: 0 15px 32px rgba(0,0,0,.25);
    }

    .sidebar-card-title {
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 950;
        margin-bottom: 7px;
    }

    .sidebar-card-text {
        color: #cbd5e1;
        font-size: .9rem;
        line-height: 1.35;
        font-weight: 650;
    }

    .stTabs [data-baseweb="tab"] {
        background: #0b1020;
        border-radius: 12px 12px 0 0;
        color: #dbeafe;
        font-weight: 850;
        border: 1px solid rgba(96,165,250,.18);
    }

    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: #172033 !important;
        border-bottom: 2px solid #60a5fa !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(96,165,250,.18);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 18px 38px rgba(0,0,0,.28);
    }
</style>
""", unsafe_allow_html=True)


def clamp(x, low=0, high=100):
    try:
        return max(low, min(high, float(x)))
    except Exception:
        return 50


def safe_float(x, default=0):
    try:
        if x in [None, "", "-", ".---"]:
            return default
        return float(str(x).replace(",", ""))
    except Exception:
        return default


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).replace("<", "").replace(">", "")


@st.cache_data(ttl=900)
def mlb_get(endpoint, params=None):
    url = f"{MLB_BASE}/{endpoint.lstrip('/')}"
    r = requests.get(url, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()


def get_stat(stats_list, key, default=0):
    if not stats_list:
        return default
    splits = stats_list[0].get("splits", [])
    if not splits:
        return default
    stat = splits[0].get("stat", {})
    return stat.get(key, default)


@st.cache_data(ttl=3600)
def fetch_team_stats(team_id):
    out = {}
    try:
        hitting = mlb_get(f"teams/{team_id}/stats", {"stats": "season", "group": "hitting"})
        pitching = mlb_get(f"teams/{team_id}/stats", {"stats": "season", "group": "pitching"})
        h_stats = hitting.get("stats", [])
        p_stats = pitching.get("stats", [])

        out["avg"] = safe_float(get_stat(h_stats, "avg", 0))
        out["obp"] = safe_float(get_stat(h_stats, "obp", 0))
        out["slg"] = safe_float(get_stat(h_stats, "slg", 0))
        out["ops"] = safe_float(get_stat(h_stats, "ops", 0))
        out["runs"] = safe_float(get_stat(h_stats, "runs", 0))
        out["games"] = max(safe_float(get_stat(h_stats, "gamesPlayed", 1)), 1)
        out["runs_per_game"] = out["runs"] / out["games"]

        out["pitch_era"] = safe_float(get_stat(p_stats, "era", 4.50))
        out["pitch_whip"] = safe_float(get_stat(p_stats, "whip", 1.30))
        out["pitch_so"] = safe_float(get_stat(p_stats, "strikeOuts", 0))
        out["pitch_bb"] = max(safe_float(get_stat(p_stats, "baseOnBalls", 1)), 1)
        out["pitch_kbb"] = out["pitch_so"] / out["pitch_bb"]
    except Exception:
        out = {
            "avg": .245, "obp": .315, "slg": .400, "ops": .715,
            "runs_per_game": 4.3,
            "pitch_era": 4.50, "pitch_whip": 1.30, "pitch_kbb": 2.4
        }
    return out


@st.cache_data(ttl=3600)
def fetch_pitcher_stats(person_id):
    if not person_id:
        return None
    try:
        data = mlb_get(f"people/{person_id}/stats", {"stats": "season", "group": "pitching"})
        stats = data.get("stats", [])
        return {
            "era": safe_float(get_stat(stats, "era", 4.50)),
            "whip": safe_float(get_stat(stats, "whip", 1.30)),
            "so": safe_float(get_stat(stats, "strikeOuts", 0)),
            "bb": max(safe_float(get_stat(stats, "baseOnBalls", 1)), 1),
            "ip": safe_float(str(get_stat(stats, "inningsPitched", 0)).split(".")[0], 0),
            "games": safe_float(get_stat(stats, "gamesPlayed", 0))
        }
    except Exception:
        return None


@st.cache_data(ttl=900)
def fetch_schedule(slate_date):
    data = mlb_get("schedule", {
        "sportId": 1,
        "date": slate_date.isoformat(),
        "hydrate": "probablePitcher,team"
    })
    games = []
    for d in data.get("dates", []):
        games.extend(d.get("games", []))
    return games


@st.cache_data(ttl=600)
def fetch_lineup_score(game_pk, side):
    """
    Uses MLB boxscore battingOrder when available.
    If lineups are not posted, returns a neutral score and status.
    """
    try:
        data = mlb_get(f"game/{game_pk}/boxscore")
        team_data = data.get("teams", {}).get(side, {})
        players = team_data.get("players", {})
        batting_order = []
        for _, p in players.items():
            bo = p.get("battingOrder")
            if bo:
                batting_order.append((int(bo), p))
        if not batting_order:
            return 66, "Lineup not confirmed"

        batting_order = sorted(batting_order, key=lambda x: x[0])[:9]
        ops_values = []
        for _, player in batting_order:
            stat = player.get("seasonStats", {}).get("batting", {})
            ops = safe_float(stat.get("ops", 0), 0)
            if ops > 0:
                ops_values.append(ops)

        if not ops_values:
            return 72, "Confirmed lineup, limited hitter stats"

        avg_ops = sum(ops_values) / len(ops_values)
        top_six_ops = sum(ops_values[:6]) / max(len(ops_values[:6]), 1)
        score = 50 + ((avg_ops - .680) * 100) + ((top_six_ops - .700) * 75)
        return round(clamp(score, 45, 95), 1), "Confirmed lineup"
    except Exception:
        return 66, "Lineup unavailable"


def offense_score(stats):
    ops = stats.get("ops", .715)
    rpg = stats.get("runs_per_game", 4.3)
    obp = stats.get("obp", .315)
    slg = stats.get("slg", .400)

    score = (
        50
        + (ops - .700) * 85
        + (rpg - 4.30) * 5.5
        + (obp - .315) * 75
        + (slg - .400) * 45
    )
    return round(clamp(score, 35, 95), 1)


def bullpen_score(stats):
    era = stats.get("pitch_era", 4.50)
    whip = stats.get("pitch_whip", 1.30)
    kbb = stats.get("pitch_kbb", 2.4)

    score = (
        72
        + (4.50 - era) * 7.0
        + (1.30 - whip) * 25.0
        + (kbb - 2.40) * 4.0
    )
    return round(clamp(score, 35, 95), 1)


def sp_score(pitcher_stats):
    if pitcher_stats is None:
        return 62, "No probable starter stats"

    era = pitcher_stats.get("era", 4.50)
    whip = pitcher_stats.get("whip", 1.30)
    kbb = pitcher_stats.get("so", 0) / max(pitcher_stats.get("bb", 1), 1)
    ip = pitcher_stats.get("ip", 0)

    score = (
        74
        + (4.30 - era) * 7.5
        + (1.28 - whip) * 28
        + (kbb - 2.50) * 4.5
    )

    if ip < 20:
        score -= 5

    return round(clamp(score, 35, 96), 1), f"ERA {era:.2f}, WHIP {whip:.2f}, K/BB {kbb:.2f}"


def situational_score(is_home, game_number=1):
    score = 70
    if is_home:
        score += 5
    return round(clamp(score, 45, 85), 1)


def grade_score(score):
    if score >= 85:
        return "A"
    if score >= 78:
        return "B"
    if score >= 70:
        return "C"
    return "Pass"


def grade_label(score):
    grade = grade_score(score)
    if grade == "A":
        return "Core projected winner"
    if grade == "B":
        return "Playable winner candidate"
    if grade == "C":
        return "Thin / watchlist only"
    return "Pass"


def grade_class(grade):
    if grade == "A":
        return "a"
    if grade == "B":
        return "b"
    if grade == "C":
        return "c"
    return "pass"


def moneyline_to_implied(ml):
    try:
        ml = float(ml)
    except Exception:
        return None
    if math.isnan(ml) or ml == 0:
        return None
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)


def format_moneyline(ml):
    try:
        ml = float(ml)
    except Exception:
        return "—"
    if math.isnan(ml) or ml == 0:
        return "—"
    if ml > 0:
        return f"+{int(ml)}"
    return str(int(ml))


def reward_view(ml):
    try:
        ml = float(ml)
    except Exception:
        return "No odds entered"
    if math.isnan(ml) or ml == 0:
        return "No odds entered"
    if ml >= 120:
        return "Plus-money reward"
    if ml > 0:
        return "Small plus-money reward"
    if ml >= -130:
        return "Manageable favorite price"
    if ml >= -180:
        return "Expensive favorite"
    return "Very expensive favorite"


def strongest_metric(row):
    metric_names = {
        "SP_Score": "Starting Pitcher",
        "Offense_Score": "Offense",
        "Bullpen_Score": "Pitching/Bullpen",
        "Lineup_Score": "Lineup",
        "Situational_Score": "Situational"
    }
    best_col = max(metric_names.keys(), key=lambda c: float(row[c]))
    return metric_names[best_col]


def profile_flags(row):
    flags = []
    if row["SP_Score"] >= 85:
        flags.append("SP+")
    if row["Offense_Score"] >= 85:
        flags.append("OFF+")
    if row["Bullpen_Score"] >= 85:
        flags.append("PITCH+")
    if row["Lineup_Score"] >= 85:
        flags.append("LU+")
    if row["Situational_Score"] >= 78:
        flags.append("SIT+")
    return " / ".join(flags) if flags else "Balanced"


def score_board(df, weights):
    df = df.copy()
    total_weight = max(sum(weights.values()), 1)

    df["Win_Score"] = (
        df["SP_Score"] * weights["SP_Score"] +
        df["Offense_Score"] * weights["Offense_Score"] +
        df["Bullpen_Score"] * weights["Bullpen_Score"] +
        df["Lineup_Score"] * weights["Lineup_Score"] +
        df["Situational_Score"] * weights["Situational_Score"]
    ) / total_weight

    df["Win_Score"] = df["Win_Score"].round(1)
    df["Grade"] = df["Win_Score"].apply(grade_score)
    df["Suggested_Status"] = df["Win_Score"].apply(grade_label)
    df["Strongest_Metric"] = df.apply(strongest_metric, axis=1)
    df["Profile"] = df.apply(profile_flags, axis=1)
    df["Moneyline_Display"] = df["Moneyline"].apply(format_moneyline)
    df["Implied_Probability"] = df["Moneyline"].apply(moneyline_to_implied)
    df["Implied_Probability"] = df["Implied_Probability"].apply(lambda x: round(x * 100, 1) if x is not None else None)
    df["Reward_View"] = df["Moneyline"].apply(reward_view)

    df = df.sort_values("Win_Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


def build_live_board(slate_date):
    games = fetch_schedule(slate_date)
    rows = []

    for idx, game in enumerate(games, start=1):
        game_pk = game.get("gamePk")
        teams = game.get("teams", {})
        away = teams.get("away", {})
        home = teams.get("home", {})

        away_team = away.get("team", {})
        home_team = home.get("team", {})

        away_prob = away.get("probablePitcher", {})
        home_prob = home.get("probablePitcher", {})

        game_state = game.get("status", {}).get("detailedState", "")

        for side, opp_side, is_home in [
            ("away", "home", False),
            ("home", "away", True)
        ]:
            team_obj = teams.get(side, {}).get("team", {})
            opp_obj = teams.get(opp_side, {}).get("team", {})
            probable = teams.get(side, {}).get("probablePitcher", {})

            team_id = team_obj.get("id")
            opp_id = opp_obj.get("id")
            team_name = team_obj.get("abbreviation") or team_obj.get("teamName") or team_obj.get("name")
            opp_name = opp_obj.get("abbreviation") or opp_obj.get("teamName") or opp_obj.get("name")
            pitcher_name = probable.get("fullName", "TBD")
            pitcher_id = probable.get("id")

            team_stats = fetch_team_stats(team_id)
            pitcher_stats = fetch_pitcher_stats(pitcher_id)

            sp, sp_note = sp_score(pitcher_stats)
            off = offense_score(team_stats)
            pen = bullpen_score(team_stats)
            lu, lu_status = fetch_lineup_score(game_pk, side)
            sit = situational_score(is_home)

            notes = f"{sp_note}; {lu_status}; game status: {game_state}"

            rows.append({
                "Game": idx,
                "Team": team_name,
                "Opponent": opp_name,
                "Home": "Yes" if is_home else "No",
                "Team_ID": team_id,
                "Opponent_ID": opp_id,
                "Game_PK": game_pk,
                "Starting_Pitcher": pitcher_name,
                "SP_Score": sp,
                "Offense_Score": off,
                "Bullpen_Score": pen,
                "Lineup_Score": lu,
                "Situational_Score": sit,
                "Moneyline": None,
                "Notes": notes
            })

    return pd.DataFrame(rows)



def apply_one_side_per_game(board):
    """
    Keeps only the highest-rated team from each MLB game as a suggested side.
    Both teams are still available in the full board for transparency.
    """
    board = board.copy()

    if "Game_PK" not in board.columns:
        board["Suggested_Side"] = True
        board["Game_Winner_Filter"] = "Suggested"
        return board, board

    # Rank teams inside each game by Win Score.
    board["Game_Side_Rank"] = board.groupby("Game_PK")["Win_Score"].rank(
        method="first",
        ascending=False
    )

    board["Suggested_Side"] = board["Game_Side_Rank"] == 1
    board["Game_Winner_Filter"] = board["Suggested_Side"].apply(
        lambda x: "Suggested Side" if x else "Opponent Side / Not Suggested"
    )

    suggested = board[board["Suggested_Side"]].copy()
    suggested = suggested.sort_values("Win_Score", ascending=False).reset_index(drop=True)
    suggested["Rank"] = suggested.index + 1

    full = board.sort_values(["Game_PK", "Game_Side_Rank"]).reset_index(drop=True)
    return suggested, full


def render_card(row):
    g = safe_text(row["Grade"])
    cls = grade_class(g)
    home_text = "HOME" if str(row["Home"]).lower() in ["yes", "true", "1", "home"] else "AWAY"
    notes = safe_text(row.get("Notes", ""))[:95]
    team = safe_text(row["Team"])
    opp = safe_text(row["Opponent"])
    metric = safe_text(row["Strongest_Metric"])
    profile = safe_text(row["Profile"])
    ml = safe_text(row.get("Moneyline_Display", "—"))
    reward = safe_text(row.get("Reward_View", "No odds entered"))
    pitcher = safe_text(row.get("Starting_Pitcher", "TBD"))

    st.markdown(f"""
    <div class="team-card team-card-{cls}">
        <div class="rank-line">RANK #{int(row["Rank"])} · {home_text}</div>
        <div class="team-name">{team}</div>
        <div class="matchup">vs {opp}</div>
        <div class="pitcher">SP: {pitcher}</div>
        <div class="score-flex">
            <div>
                <div class="score-label">Win Score</div>
                <div class="win-score">{row["Win_Score"]}</div>
            </div>
            <div class="grade-pill pill-{cls}">{g}</div>
        </div>
        <div class="odds">Odds: {ml}</div>
        <div class="odds-note">{reward} · odds do not affect rank</div>
        <div class="footer-note"><b>{metric}</b> · {profile}<br>{notes}</div>
    </div>
    """, unsafe_allow_html=True)


def render_team_tiles(board, max_cards=8):
    top = board.head(max_cards).reset_index(drop=True)
    for start in range(0, len(top), 4):
        cols = st.columns(4)
        chunk = top.iloc[start:start+4]
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                render_card(row)


def color_grade(val):
    if val == "A":
        return "background-color: #123d22; color: #7ee787; font-weight: 800"
    if val == "B":
        return "background-color: #102f4f; color: #bfdbfe; font-weight: 800"
    if val == "C":
        return "background-color: #3d2f10; color: #fde68a; font-weight: 800"
    if val == "Pass":
        return "background-color: #4a1717; color: #fecaca; font-weight: 800"
    return ""


def style_grades(df):
    return df.style.map(color_grade, subset=["Grade"])


st.markdown("""
<div class="hero-card">
    <div class="hero-title">⚾ Moneyline Winners</div>
    <div class="hero-subtitle">
        Live MLB projected winner engine. Pulls today’s slate and suggests only one side per game.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-strip">
    Suggested picks are based only on Win Score and Grade. Moneyline odds are optional reward context and never change the picks.
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Live Engine</div>
        <div class="sidebar-card-text">
            Pulls schedule, probable starters, team offense, team pitching, and available lineup data from MLB.
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected_date = st.date_input("Slate Date", value=date.today())

    if st.button("Refresh Live MLB Data", use_container_width=True):
        st.cache_data.clear()
        with st.spinner("Pulling MLB slate and scoring teams..."):
            try:
                st.session_state["live_data"] = build_live_board(selected_date)
                st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
                st.success("Live data refreshed.")
            except Exception as e:
                st.error(f"Live refresh failed: {e}")

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Scoring Weights</div>
        <div class="sidebar-card-text">
            Projected win likelihood only. Odds do not affect these scores.
        </div>
    </div>
    """, unsafe_allow_html=True)

    weights = {
        "SP_Score": st.number_input("Starting Pitcher", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["SP_Score"]),
        "Offense_Score": st.number_input("Offense", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["Offense_Score"]),
        "Bullpen_Score": st.number_input("Pitching/Bullpen", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["Bullpen_Score"]),
        "Lineup_Score": st.number_input("Lineup", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["Lineup_Score"]),
        "Situational_Score": st.number_input("Situational", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["Situational_Score"]),
    }

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Decision Rules</div>
        <div class="sidebar-card-text">
            <b>85+</b> = A — Core projected winner<br>
            <b>78-84</b> = B — Playable<br>
            <b>70-77</b> = C — Thin<br>
            <b>&lt;70</b> = Pass
        </div>
    </div>
    """, unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Data / Odds Input", "Raw Engine Data", "Metric Guide"])

if "live_data" not in st.session_state:
    try:
        with st.spinner("Loading today's MLB slate..."):
            st.session_state["live_data"] = build_live_board(selected_date)
            st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    except Exception as e:
        st.session_state["live_data"] = pd.DataFrame(columns=ENGINE_COLUMNS)
        st.session_state["load_error"] = str(e)

with tab2:
    st.subheader("Optional Odds / Manual Adjustments")
    st.write("The engine auto-generates scores. You can optionally enter moneylines or adjust a score if needed.")

    if "load_error" in st.session_state:
        st.error(f"Initial live load error: {st.session_state['load_error']}")

    df_edit = st.session_state["live_data"].copy()

    edited = st.data_editor(
        df_edit,
        num_rows="fixed",
        use_container_width=True,
        key="engine_editor",
        column_config={
            "Moneyline": st.column_config.NumberColumn("Moneyline", help="Optional. Does not affect rank."),
            "SP_Score": st.column_config.NumberColumn("SP Score", min_value=0, max_value=100),
            "Offense_Score": st.column_config.NumberColumn("Offense Score", min_value=0, max_value=100),
            "Bullpen_Score": st.column_config.NumberColumn("Pitching/Bullpen Score", min_value=0, max_value=100),
            "Lineup_Score": st.column_config.NumberColumn("Lineup Score", min_value=0, max_value=100),
            "Situational_Score": st.column_config.NumberColumn("Situational Score", min_value=0, max_value=100),
        }
    )
    st.session_state["live_data"] = edited

    st.download_button(
        "Export Current Engine Data",
        data=edited.to_csv(index=False),
        file_name=f"moneyline_winners_engine_data_{selected_date.isoformat()}.csv",
        mime="text/csv"
    )

with tab1:
    df = st.session_state["live_data"].copy()

    if df.empty:
        st.error("No MLB games loaded. Try Refresh Live MLB Data.")
    else:
        raw_board = score_board(df, weights)
        suggested_board, full_board = apply_one_side_per_game(raw_board)

        a_count = int((suggested_board["Grade"] == "A").sum())
        b_count = int((suggested_board["Grade"] == "B").sum())
        top_score = suggested_board["Win_Score"].max() if len(suggested_board) else 0
        confirmed_lineups = int(raw_board["Notes"].str.contains("Confirmed lineup", na=False).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Games Ranked", len(suggested_board))
        m2.metric("A-Grade Sides", a_count)
        m3.metric("B-Grade Sides", b_count)
        m4.metric("Top Score", top_score)

        st.caption(f"Last refresh: {st.session_state.get('last_refresh', 'Not refreshed yet')}")

        st.markdown("""
        <div class="info-strip">
            One-side-per-game rule is active: the suggested board only shows the highest-rated team from each matchup.
        </div>
        """, unsafe_allow_html=True)

        if confirmed_lineups < len(raw_board):
            st.markdown(f"""
            <div class="warning-strip">
                {confirmed_lineups} of {len(raw_board)} team lineups appear confirmed from MLB boxscore data. 
                Lineup Score is automatically neutral until lineups are available.
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Top Projected Winners</div>', unsafe_allow_html=True)
        render_team_tiles(suggested_board, max_cards=8)

        st.markdown('<div class="section-title">Suggested Winners Board</div>', unsafe_allow_html=True)
        suggested_cols = [
            "Rank", "Team", "Opponent", "Home", "Starting_Pitcher", "Win_Score", "Grade",
            "Suggested_Status", "Moneyline_Display", "Implied_Probability",
            "Reward_View", "SP_Score", "Offense_Score", "Bullpen_Score",
            "Lineup_Score", "Situational_Score", "Strongest_Metric", "Profile", "Notes"
        ]

        st.dataframe(
            style_grades(suggested_board[suggested_cols]),
            use_container_width=True,
            hide_index=True
        )

        st.markdown('<div class="section-title">Full Matchup Comparison</div>', unsafe_allow_html=True)
        full_cols = [
            "Game_PK", "Game_Winner_Filter", "Team", "Opponent", "Home", "Starting_Pitcher",
            "Win_Score", "Grade", "Moneyline_Display", "SP_Score", "Offense_Score",
            "Bullpen_Score", "Lineup_Score", "Situational_Score", "Notes"
        ]

        st.dataframe(
            style_grades(full_board[full_cols]),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Export Suggested Winners Board",
            data=suggested_board.to_csv(index=False),
            file_name=f"moneyline_winners_suggested_board_{selected_date.isoformat()}.csv",
            mime="text/csv"
        )

        st.download_button(
            "Export Full Matchup Comparison",
            data=full_board.to_csv(index=False),
            file_name=f"moneyline_winners_full_matchups_{selected_date.isoformat()}.csv",
            mime="text/csv"
        )

with tab3:
    st.subheader("Raw Engine Data")
    st.write("These are the auto-generated category scores before final weighting.")
    st.dataframe(st.session_state["live_data"], use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Metric Guide")

    guide = pd.DataFrame([
        ["Starting Pitcher", "MLB probable starter season ERA, WHIP, K/BB, innings sample", "40%"],
        ["Offense", "MLB team season OPS, OBP, SLG, runs per game", "25%"],
        ["Pitching/Bullpen", "MLB team pitching ERA, WHIP, K/BB as current proxy", "15%"],
        ["Lineup", "MLB boxscore battingOrder when available; otherwise neutral placeholder", "12%"],
        ["Situational", "Home field baseline; can be expanded with travel/rest/weather", "8%"],
        ["Moneyline", "Optional reward context only. Does not affect rank or suggested status.", "0%"],
    ], columns=["Category", "Current Live Engine Input", "Model Weight"])

    st.dataframe(guide, use_container_width=True, hide_index=True)

    st.warning(
        "This is the first real live engine. Next refinements should replace team pitching proxy with true bullpen data, add handedness splits, add last-14-day form, and connect optional Odds API."
    )
