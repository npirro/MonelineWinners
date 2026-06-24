
import streamlit as st
import pandas as pd
from datetime import date
import math

st.set_page_config(
    page_title="Moneyline Winners",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

CORE_COLUMNS = [
    "Game", "Team", "Opponent", "Home",
    "SP_Score", "Offense_Score", "Bullpen_Score",
    "Lineup_Score", "Situational_Score", "Notes"
]

OPTIONAL_COLUMNS = ["Moneyline"]
ALL_COLUMNS = CORE_COLUMNS + OPTIONAL_COLUMNS

DEFAULT_WEIGHTS = {
    "SP_Score": 40,
    "Offense_Score": 25,
    "Bullpen_Score": 15,
    "Lineup_Score": 12,
    "Situational_Score": 8
}

SAMPLE_DATA = pd.DataFrame([
    [1, "PHI", "NYM", "Yes", 88, 84, 77, 86, 72, "Strong SP and confirmed lineup", -155],
    [1, "NYM", "PHI", "No", 71, 78, 69, 75, 64, "Bullpen concern", 135],
    [2, "LAD", "SF", "Yes", 91, 89, 82, 88, 78, "Best full-game profile", -210],
    [2, "SF", "LAD", "No", 66, 72, 74, 69, 60, "Tough matchup", 180],
    [3, "NYY", "BOS", "No", 84, 86, 76, 81, 68, "Offense advantage", -120],
    [3, "BOS", "NYY", "Yes", 73, 80, 71, 77, 74, "Home field helps", 105],
    [4, "SEA", "TEX", "Yes", 79, 75, 88, 76, 76, "Bullpen edge", -108],
    [4, "TEX", "SEA", "No", 74, 82, 70, 79, 65, "Lineup okay but pitching gap", -102],
], columns=ALL_COLUMNS)


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

    [data-testid="stSidebar"] > div {
        padding-top: 1.1rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #f8fafc !important;
    }

    [data-testid="stSidebar"] .stCaptionContainer {
        color: #cbd5e1 !important;
        font-weight: 650;
    }

    [data-testid="stNumberInput"] {
        background: linear-gradient(145deg, #0b1020, #111827);
        border: 1px solid rgba(96,165,250,0.18);
        border-radius: 18px;
        padding: 12px 12px 10px;
        box-shadow: 0 14px 30px rgba(0,0,0,.24);
        margin-bottom: 12px;
    }

    [data-testid="stNumberInput"] label p {
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
    div[data-testid="stMetricLabel"] label,
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
        min-height: 245px;
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
        margin-bottom: 16px;
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

    .footer-note b {
        color: #ffffff;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
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

    /* KPI label fallback — keeps the top metric words readable on Streamlit Cloud */
    [data-testid="stMetric"] [data-testid="stMetricLabel"] * {
        color: #bfdbfe !important;
        opacity: 1 !important;
        font-weight: 950 !important;
    }

</style>
""", unsafe_allow_html=True)


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).replace("<", "").replace(">", "")


def grade_score(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 78:
        return "B"
    if score >= 70:
        return "C"
    return "Pass"


def grade_label(score: float) -> str:
    grade = grade_score(score)
    if grade == "A":
        return "Core projected winner"
    if grade == "B":
        return "Playable winner candidate"
    if grade == "C":
        return "Thin / watchlist only"
    return "Pass"


def grade_class(grade: str) -> str:
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


def strongest_metric(row: pd.Series) -> str:
    metric_names = {
        "SP_Score": "Starting Pitcher",
        "Offense_Score": "Offense",
        "Bullpen_Score": "Bullpen",
        "Lineup_Score": "Lineup",
        "Situational_Score": "Situational"
    }
    best_col = max(metric_names.keys(), key=lambda c: float(row[c]))
    return metric_names[best_col]


def profile_flags(row: pd.Series) -> str:
    flags = []
    if row["SP_Score"] >= 85:
        flags.append("SP+")
    if row["Offense_Score"] >= 85:
        flags.append("OFF+")
    if row["Bullpen_Score"] >= 85:
        flags.append("BP+")
    if row["Lineup_Score"] >= 85:
        flags.append("LU+")
    if row["Situational_Score"] >= 80:
        flags.append("SIT+")
    return " / ".join(flags) if flags else "Balanced"


def validate_data(df: pd.DataFrame):
    errors = []
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return errors

    score_cols = ["SP_Score", "Offense_Score", "Bullpen_Score", "Lineup_Score", "Situational_Score"]
    for col in score_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            errors.append(f"{col} contains blank or non-numeric values.")
        if ((converted < 0) | (converted > 100)).any():
            errors.append(f"{col} must be between 0 and 100.")
    return errors


def score_board(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df.copy()
    if "Moneyline" not in df.columns:
        df["Moneyline"] = None

    score_cols = ["SP_Score", "Offense_Score", "Bullpen_Score", "Lineup_Score", "Situational_Score"]
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

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


def render_card(row):
    g = safe_text(row["Grade"])
    cls = grade_class(g)
    home_text = "HOME" if str(row["Home"]).lower() in ["yes", "true", "1", "home"] else "AWAY"
    notes = safe_text(row.get("Notes", ""))[:85]
    team = safe_text(row["Team"])
    opp = safe_text(row["Opponent"])
    metric = safe_text(row["Strongest_Metric"])
    profile = safe_text(row["Profile"])
    ml = safe_text(row.get("Moneyline_Display", "—"))
    reward = safe_text(row.get("Reward_View", "No odds entered"))

    st.markdown(f"""
    <div class="team-card team-card-{cls}">
        <div class="rank-line">RANK #{int(row["Rank"])} · {home_text}</div>
        <div class="team-name">{team}</div>
        <div class="matchup">vs {opp}</div>
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


def render_team_tiles(board: pd.DataFrame, max_cards: int = 8):
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


def style_grades(df: pd.DataFrame):
    return df.style.map(color_grade, subset=["Grade"])


st.markdown("""
<div class="hero-card">
    <div class="hero-title">⚾ Moneyline Winners</div>
    <div class="hero-subtitle">
        Projected MLB winner rankings based on baseball metrics only. Odds are optional reward context and never change the picks.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-strip">
    Suggested picks are based only on Win Score and Grade. Moneyline odds are shown only so you can decide whether the reward is worth betting.
</div>
""", unsafe_allow_html=True)


with st.sidebar:
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
        "Bullpen_Score": st.number_input("Bullpen", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["Bullpen_Score"]),
        "Lineup_Score": st.number_input("Confirmed Lineup", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["Lineup_Score"]),
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


tab1, tab2, tab3 = st.tabs(["Dashboard", "Data Input", "Metric Guide"])

if "daily_data" not in st.session_state:
    st.session_state["daily_data"] = SAMPLE_DATA.copy()

with tab2:
    st.subheader("Daily Data Input")
    st.write("Moneyline is optional. It is displayed for reward context only and does not change rankings.")

    uploaded_file = st.file_uploader("Upload daily CSV", type=["csv"])

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Load Sample Slate", use_container_width=True):
            st.session_state["daily_data"] = SAMPLE_DATA.copy()
            st.success("Sample slate loaded.")
    with c2:
        if st.button("Clear Data", use_container_width=True):
            st.session_state["daily_data"] = pd.DataFrame(columns=ALL_COLUMNS)
            st.success("Data cleared.")
    with c3:
        st.download_button(
            "Download Template",
            data=pd.DataFrame(columns=ALL_COLUMNS).to_csv(index=False),
            file_name="moneyline_winners_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if "Moneyline" not in uploaded_df.columns:
                uploaded_df["Moneyline"] = None
            st.session_state["daily_data"] = uploaded_df
            st.success("CSV uploaded.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    edited = st.data_editor(
        st.session_state["daily_data"],
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )
    st.session_state["daily_data"] = edited


with tab1:
    df = st.session_state.get("daily_data", SAMPLE_DATA.copy())
    errors = validate_data(df.copy())

    if errors:
        st.error("Data validation failed.")
        for error in errors:
            st.write(f"- {error}")
    else:
        board = score_board(df, weights)

        a_count = int((board["Grade"] == "A").sum())
        b_count = int((board["Grade"] == "B").sum())
        top_score = board["Win_Score"].max() if len(board) else 0
        plus_money_count = int(pd.to_numeric(board["Moneyline"], errors="coerce").fillna(-9999).gt(0).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Teams Ranked", len(board))
        m2.metric("A-Grade Teams", a_count)
        m3.metric("B-Grade Teams", b_count)
        m4.metric("Top Score", top_score)

        st.markdown('<div class="section-title">Top Projected Winners</div>', unsafe_allow_html=True)
        render_team_tiles(board, max_cards=8)

        st.markdown('<div class="section-title">Full Projected Winners Board</div>', unsafe_allow_html=True)
        display_cols = [
            "Rank", "Team", "Opponent", "Home", "Win_Score", "Grade",
            "Suggested_Status", "Moneyline_Display", "Implied_Probability",
            "Reward_View", "Strongest_Metric", "Profile", "Notes"
        ]

        st.dataframe(
            style_grades(board[display_cols]),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Export Ranked Board",
            data=board.to_csv(index=False),
            file_name=f"moneyline_winners_board_{date.today().isoformat()}.csv",
            mime="text/csv"
        )


with tab3:
    st.subheader("Metric Guide")
    guide = pd.DataFrame([
        ["Starting Pitcher", "ERA, FIP/xFIP, WHIP, K-BB%, pitch count, handedness fit", "40%"],
        ["Offense", "wRC+, OPS, ISO, K%, BB%, recent form, split vs starter handedness", "25%"],
        ["Bullpen", "Season strength, last 14 days, rest/fatigue, leverage arms availability", "15%"],
        ["Lineup", "Confirmed lineup quality, missing bats, catcher/rest days, top-6 strength", "12%"],
        ["Situational", "Home field, travel, rest, weather, getaway spot, series context", "8%"],
        ["Moneyline", "Optional reward context only. Does not affect Win Score, rank, grade, or suggested status.", "0%"],
    ], columns=["Category", "What It Measures", "Model Weight"])

    st.dataframe(guide, use_container_width=True, hide_index=True)

    st.info("Next upgrade: connect live MLB data feeds and optional Odds API while keeping odds out of the projection score.")
