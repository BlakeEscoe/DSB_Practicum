import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from nflreadpy.ui import (
        SearchConfig,
        answer_query,
        load_recent_player_stats,
        parse_compare_names,
        parse_seasons,
        _apply_split_filters,
        SPLIT_CATEGORIES,
    )
except Exception as exc:
    st.error(f"Unable to import nflreadpy UI helpers: {exc}")
    st.stop()

from sleeper_connect import get_user_id, get_leagues, get_all_rosters

st.set_page_config(page_title="Fantasy Football Stat Explorer", layout="wide")

st.title("NFL Stat Finder")
st.markdown(
    "Search recent nflreadpy player stats with splits, comparisons, and Sleeper league integration."
)

EXAMPLE_QUERIES = [
    "Patrick Mahomes passing yards",
    "rushing yards leaders",
    "Ja'Marr Chase fantasy points PPR",
    "Lions away rushing yards",
    "Justin Jefferson receiving yards",
]

@st.cache_data(show_spinner=False)
def load_stats(seasons_text: str):
    seasons = parse_seasons(seasons_text)
    return load_recent_player_stats(seasons)


def render_result(answer: dict[str, any]) -> None:
    if not answer:
        st.warning("No result returned.")
        return

    st.subheader(answer.get("title", "Result"))
    st.write(answer.get("summary", ""))

    projection = answer.get("projection")
    if projection:
        with st.expander("Projection details", expanded=True):
            st.metric("Projection", projection.get("projection"))
            st.metric("Recent average", projection.get("recent_average"))
            st.metric("Sample average", projection.get("sample_average"))
            st.metric("Trend", projection.get("direction"))
            st.write(projection.get("method"))

    if answer.get("type") == "split_summary":
        rows = answer.get("rows", [])
        if rows:
            st.write("#### Split summary")
            st.dataframe(rows)
        return

    rows = answer.get("rows", [])
    if rows:
        st.dataframe(rows)
    else:
        st.info("No rows available for this query.")


def run_search():
    with st.form("search_form"):
        left, right = st.columns([3, 2])
        with left:
            query = st.text_input("Player or team query", value="")
            compare = st.text_input(
                "Compare players",
                help="Enter 2-3 player names separated by commas, and, or vs.",
            )
            seasons = st.text_input(
                "Seasons",
                value=", ".join(str(season) for season in parse_seasons(None)),
                help="Comma-separated season list",
            )
        with right:
            split_category = st.selectbox(
                "Split summary category",
                [""] + list(SPLIT_CATEGORIES.keys()),
                format_func=lambda value: "None" if value == "" else SPLIT_CATEGORIES[value][1],
            )
            location = st.selectbox("Location", ["", "home", "away"])
            roof = st.selectbox("Roof", ["", "indoors", "outdoors"])
            day = st.selectbox(
                "Day", ["", "weekday", "weekend", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            )
            time_of_day = st.selectbox(
                "Time of day", ["", "europe", "early", "midday", "night"]
            )
            weather = st.selectbox("Weather", ["", "rain", "snow", "below_0"])
            divisional = st.checkbox("Divisional rivals only")

        submitted = st.form_submit_button("Search")

    if submitted:
        if not query and not compare and not split_category:
            st.warning("Enter a query, compare players, or select a split summary.")
            return

        try:
            df = load_stats(seasons)
        except Exception as exc:
            st.error(f"Failed to load stats: {exc}")
            return

        params = {
            "location": [location],
            "roof": [roof],
            "day": [day],
            "time_of_day": [time_of_day],
            "weather": [weather],
            "divisional": ["1" if divisional else ""],
        }

        df = _apply_split_filters(df, params)
        compare_names = parse_compare_names(compare)
        split_value = split_category or None
        answer = answer_query(
            df,
            SearchConfig(
                query=query,
                seasons=parse_seasons(seasons),
                limit=12,
                compare=compare_names,
                split_category=split_value,
            ),
        )
        render_result(answer)


with st.expander("Search examples", expanded=False):
    st.write(", ".join(EXAMPLE_QUERIES))

run_search()

st.markdown("---")

st.header("Sleeper league lookup")

username = st.text_input("Sleeper username for league lookup", value="")
season = st.selectbox("Sleeper season", [2026, 2025, 2024], index=0, key="sleeper_season")

if st.button("Find My Leagues", key="find_leagues"):
    if not username:
        st.warning("Enter your Sleeper username.")
    else:
        user_id = get_user_id(username)
        if isinstance(user_id, str) and user_id.startswith("Error"):
            st.error(user_id)
        else:
            st.success(f"User ID found: {user_id}")
            leagues = get_leagues(user_id, season)
            if isinstance(leagues, str) and leagues.startswith("Error"):
                st.error(leagues)
            elif not leagues:
                st.warning("No leagues found for this user and season.")
            else:
                st.session_state["leagues"] = leagues

if "leagues" in st.session_state:
    leagues = st.session_state["leagues"]
    selected_league_name = st.selectbox("Select a league", list(leagues.keys()), key="selected_league_name")
    selected_league_id = leagues[selected_league_name]
    st.write("Selected League ID:", selected_league_id)
    if st.button("View Rosters", key="view_rosters"):
        rosters = get_all_rosters(selected_league_id)
        if isinstance(rosters, str) and rosters.startswith("Error"):
            st.error(rosters)
        else:
            st.subheader("League Rosters")
            st.json(rosters)
