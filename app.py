
import streamlit as st
import pandas as pd
from datetime import date
import math

st.set_page_config(
    page_title="Moneyline Winners",
    page_icon="⚾",
    layout="wide"
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
    .block-container { padding-top: 2rem; max-width: 1500px; }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #111827, #1f2937);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 18px 20px;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }

    div[data-testid="stMetricLabel"] { color: #9ca3af; font-weight: 700; }
    div[data-testid="stMetricValue"] { font-size: 2.1rem; font-weight: 900; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #1e293b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 28px 30px;
        margin-bottom: 18px;
        box-shadow: 0 18px 40px rgba(0,0,0,0.25);
    }

    .hero h1 { font-size: 3rem; margin-bottom: 4px; }
    .hero p { color: #9ca3af; font-size: 1.05rem; }

    .tile-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
        margin-top: 12px;
        margin-bottom: 22px;
    }

    .team-card {
        background: linear-gradient(145deg, #111827, #1f2937);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 14px 32px rgba(0,0,0,0.22);
        min-height: 205px;
    }

    .team-card-a { border-left: 7px solid #22c55e; }
    .team-card-b { border-left: 7px solid #3b82f6; }
    .team-card-c { border-left: 7px solid #f59e0b; }
    .team-card-pass { border-left: 7px solid #ef4444; }

    .rank {
        color: #9ca3af;
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .team-name {
        font-size: 2rem;
        font-weight: 950;
        margin-top: 4px;
        margin-bottom: 0px;
    }

    .matchup {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 14px;
    }

    .score-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
    }

    .win-score {
        font-size: 2rem;
        font-weight: 950;
    }

    .grade-pill {
        border-radius: 999px;
        padding: 6px 12px;
        font-weight: 900;
        font-size: .85rem;
    }

    .pill-a { background: rgba(34,197,94,.16); color: #86efac; }
    .pill-b { background: rgba(59,130,246,.16); color: #93c5fd; }
    .pill-c { background: rgba(245,158,11,.16); color: #fcd34d; }
    .pill-pass { background: rgba(239,68,68,.16); color: #fca5a5; }

    .odds-line {
        margin-top: 10px;
        color: #e5e7eb;
        font-weight: 850;
        font-size: .95rem;
    }

    .odds-caption {
        color: #9ca3af;
        font-size: .78rem;
        margin-top: 2px;
    }

    .card-footer {
        color: #9ca3af;
        font-size: .85rem;
        margin-top: 12px;
        line-height: 1.35;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 900;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .note-box {
        background: rgba(59,130,246,.10);
        border: 1px solid rgba(59,130,246,.25);
        color: #bfdbfe;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 18px;
        font-weight: 650;
    }

    @media (max-width: 1200px) {
        .tile-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 700px) {
        .tile-grid { grid-template-columns: 1fr; }
        .hero h1 { font-size: 2.2rem; }
    }
</style>
""", unsafe_allow_html=True)


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
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            errors.append(f"{col} contains blank or non-numeric values.")
        if ((df[col] < 0) | (df[col] > 100)).any():
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


def render_team_tiles(board: pd.DataFrame, max_cards: int = 8):
    cards = []
    for _, row in board.head(max_cards).iterrows():
        g = row["Grade"]
        cls = grade_class(g)
        home_text = "Home" if str(row["Home"]).lower() in ["yes", "true", "1", "home"] else "Away"
        notes = str(row.get("Notes", ""))[:80]
        ml = row.get("Moneyline_Display", "—")
        reward = row.get("Reward_View", "No odds entered")

        cards.append(f"""
        <div class="team-card team-card-{cls}">
            <div class="rank">Rank #{int(row["Rank"])} · {home_text}</div>
            <div class="team-name">{row["Team"]}</div>
            <div class="matchup">vs {row["Opponent"]}</div>
            <div class="score-row">
                <div>
                    <div style="color:#9ca3af;font-size:.8rem;font-weight:800;">WIN SCORE</div>
                    <div class="win-score">{row["Win_Score"]}</div>
                </div>
                <div class="grade-pill pill-{cls}">{g}</div>
            </div>
            <div class="odds-line">Odds: {ml}</div>
            <div class="odds-caption">{reward} · odds do not affect rank</div>
            <div class="card-footer">
                <b>{row["Strongest_Metric"]}</b> · {row["Profile"]}<br>
                {notes}
            </div>
        </div>
        """)

    st.markdown('<div class="tile-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def color_grade(val):
    if val == "A":
        return "background-color: #123d22; color: #7ee787; font-weight: 800"
    if val == "B":
        return "background-color: #102f4f; color: #79c0ff; font-weight: 800"
    if val == "C":
        return "background-color: #3d2f10; color: #e3b341; font-weight: 800"
    if val == "Pass":
        return "background-color: #4a1717; color: #ff7b72; font-weight: 800"
    return ""


def style_grades(df: pd.DataFrame):
    return df.style.map(color_grade, subset=["Grade"])


st.markdown("""
<div class="hero">
    <h1>⚾ Moneyline Winners</h1>
    <p>Projected MLB winner rankings based on baseball metrics only. Odds are optional and informational only.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="note-box">
    Suggested picks are based only on Win Score and Grade. Moneyline odds are shown only so you can decide whether the reward is worth betting.
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Scoring Weights")
    st.caption("Projected win likelihood only. Odds do not affect these scores.")

    weights = {
        "SP_Score": st.number_input("Starting Pitcher", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["SP_Score"]),
        "Offense_Score": st.number_input("Offense", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["Offense_Score"]),
        "Bullpen_Score": st.number_input("Bullpen", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["Bullpen_Score"]),
        "Lineup_Score": st.number_input("Confirmed Lineup", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["Lineup_Score"]),
        "Situational_Score": st.number_input("Situational", min_value=0, max_value=100, value=DEFAULT_WEIGHTS["Situational_Score"]),
    }

    st.divider()
    st.header("Decision Rules")
    st.write("**85+** = A — Core projected winner")
    st.write("**78-84** = B — Playable")
    st.write("**70-77** = C — Thin")
    st.write("**<70** = Pass")

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
