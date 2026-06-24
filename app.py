
import streamlit as st
import pandas as pd
from datetime import date

st.set_page_config(
    page_title="Moneyline Winners",
    page_icon="⚾",
    layout="wide"
)

REQUIRED_COLUMNS = [
    "Game", "Team", "Opponent", "Home",
    "SP_Score", "Offense_Score", "Bullpen_Score",
    "Lineup_Score", "Situational_Score", "Notes"
]

DEFAULT_WEIGHTS = {
    "SP_Score": 40,
    "Offense_Score": 25,
    "Bullpen_Score": 15,
    "Lineup_Score": 12,
    "Situational_Score": 8
}

SAMPLE_DATA = pd.DataFrame([
    [1, "PHI", "NYM", "Yes", 88, 84, 77, 86, 72, "Strong SP and confirmed lineup"],
    [1, "NYM", "PHI", "No", 71, 78, 69, 75, 64, "Bullpen concern"],
    [2, "LAD", "SF", "Yes", 91, 89, 82, 88, 78, "Best full-game profile"],
    [2, "SF", "LAD", "No", 66, 72, 74, 69, 60, "Tough matchup"],
    [3, "NYY", "BOS", "No", 84, 86, 76, 81, 68, "Offense advantage"],
    [3, "BOS", "NYY", "Yes", 73, 80, 71, 77, 74, "Home field helps"],
    [4, "SEA", "TEX", "Yes", 79, 75, 88, 76, 76, "Bullpen edge"],
    [4, "TEX", "SEA", "No", 74, 82, 70, 79, 65, "Lineup okay but pitching gap"],
], columns=REQUIRED_COLUMNS)


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
        return "A — Core projected winner"
    if grade == "B":
        return "B — Playable winner candidate"
    if grade == "C":
        return "C — Thin / watchlist only"
    return "Pass"


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
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
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
    total_weight = sum(weights.values())

    df["Win_Score"] = (
        df["SP_Score"] * weights["SP_Score"] +
        df["Offense_Score"] * weights["Offense_Score"] +
        df["Bullpen_Score"] * weights["Bullpen_Score"] +
        df["Lineup_Score"] * weights["Lineup_Score"] +
        df["Situational_Score"] * weights["Situational_Score"]
    ) / total_weight

    df["Win_Score"] = df["Win_Score"].round(1)
    df["Grade"] = df["Win_Score"].apply(grade_score)
    df["Decision"] = df["Win_Score"].apply(grade_label)
    df["Strongest_Metric"] = df.apply(strongest_metric, axis=1)
    df["Profile"] = df.apply(profile_flags, axis=1)

    df = df.sort_values("Win_Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


def style_grades(df: pd.DataFrame):
    def grade_color(val):
        if val == "A":
            return "background-color: #123d22; color: #7ee787; font-weight: 800"
        if val == "B":
            return "background-color: #102f4f; color: #79c0ff; font-weight: 800"
        if val == "C":
            return "background-color: #3d2f10; color: #e3b341; font-weight: 800"
        if val == "Pass":
            return "background-color: #4a1717; color: #ff7b72; font-weight: 800"
        return ""

    return df.style.applymap(grade_color, subset=["Grade"])


st.title("⚾ Moneyline Winners")
st.caption("MLB projected winner rankings based on baseball metrics only. No odds. No EV. No parlays.")

with st.sidebar:
    st.header("Scoring Weights")
    st.caption("Default model is built around projected win likelihood, not betting value.")

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
    st.write("**78-84** = B — Playable winner candidate")
    st.write("**70-77** = C — Thin / watchlist only")
    st.write("**<70** = Pass")

tab1, tab2, tab3 = st.tabs(["Dashboard", "Data Input", "Metric Guide"])

with tab2:
    st.subheader("Daily Data Input")
    st.write("Upload a CSV or use the sample slate. Scores should be 0-100.")

    uploaded_file = st.file_uploader("Upload daily CSV", type=["csv"])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        use_sample = st.button("Load Sample Slate")
    with col_b:
        clear_data = st.button("Clear Session Data")

    if clear_data:
        st.session_state.pop("daily_data", None)
        st.success("Session data cleared.")

    if uploaded_file is not None:
        try:
            st.session_state["daily_data"] = pd.read_csv(uploaded_file)
            st.success("CSV uploaded.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if use_sample or "daily_data" not in st.session_state:
        if use_sample:
            st.session_state["daily_data"] = SAMPLE_DATA.copy()
            st.success("Sample slate loaded.")
        elif "daily_data" not in st.session_state:
            st.session_state["daily_data"] = SAMPLE_DATA.copy()

    editable_df = st.data_editor(
        st.session_state["daily_data"],
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )

    st.session_state["daily_data"] = editable_df

    st.download_button(
        "Download Blank CSV Template",
        data=pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(index=False),
        file_name="moneyline_winners_template.csv",
        mime="text/csv"
    )

with tab1:
    df = st.session_state.get("daily_data", SAMPLE_DATA.copy())
    errors = validate_data(df.copy())

    if errors:
        st.error("Data validation failed.")
        for error in errors:
            st.write(f"- {error}")
    else:
        board = score_board(df, weights)

        top_score = board["Win_Score"].max() if len(board) else 0
        a_count = int((board["Grade"] == "A").sum())
        b_count = int((board["Grade"] == "B").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Teams Ranked", len(board))
        m2.metric("A-Grade Teams", a_count)
        m3.metric("B-Grade Teams", b_count)
        m4.metric("Top Score", top_score)

        st.subheader("Projected Winners Board")

        display_cols = [
            "Rank", "Team", "Opponent", "Home", "Win_Score", "Grade",
            "Decision", "Strongest_Metric", "Profile", "Notes"
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

        st.subheader("A-Grade / B-Grade Shortlist")
        shortlist = board[board["Grade"].isin(["A", "B"])][display_cols]
        if len(shortlist):
            st.dataframe(shortlist, use_container_width=True, hide_index=True)
        else:
            st.warning("No A-grade or B-grade teams on this slate.")

with tab3:
    st.subheader("Metric Guide")
    st.write("The goal is to rank teams by likelihood of winning, not by sportsbook value.")

    guide = pd.DataFrame([
        ["Starting Pitcher", "ERA, FIP/xFIP, WHIP, K-BB%, pitch count, handedness fit", "40%"],
        ["Offense", "wRC+, OPS, ISO, K%, BB%, recent form, split vs starter handedness", "25%"],
        ["Bullpen", "Season strength, last 14 days, rest/fatigue, leverage arms availability", "15%"],
        ["Lineup", "Confirmed lineup quality, missing bats, catcher/rest days, top-6 strength", "12%"],
        ["Situational", "Home field, travel, rest, weather, getaway spot, series context", "8%"],
    ], columns=["Category", "What It Measures", "Default Weight"])

    st.dataframe(guide, use_container_width=True, hide_index=True)

    st.info(
        "Next upgrade: connect live MLB data feeds so these category scores are generated automatically."
    )
