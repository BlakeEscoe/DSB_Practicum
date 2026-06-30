import streamlit as st
from sleeper_connect import get_user_id, get_leagues, get_all_rosters

st.set_page_config(page_title="Fantasy Football Predictor", layout="wide")

st.title("Fantasy Football Expected Points Predictor")

st.write(
    "Enter your Sleeper username to find your fantasy leagues and view roster data."
)

username = st.text_input("Sleeper Username")

season = st.selectbox("Season", [2026, 2025, 2024], index=0)

if st.button("Find My Leagues"):
    if not username:
        st.warning("Please enter a Sleeper username.")
    else:
        user_id = get_user_id(username)

        if isinstance(user_id, str) and user_id.startswith("Error"):
            st.error(user_id)
        else:
            st.session_state["user_id"] = user_id
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

    selected_league_name = st.selectbox("Select a league", list(leagues.keys()))

    selected_league_id = leagues[selected_league_name]

    st.write("Selected League ID:", selected_league_id)

    if st.button("View Rosters"):
        rosters = get_all_rosters(selected_league_id)

        if isinstance(rosters, str) and rosters.startswith("Error"):
            st.error(rosters)
        else:
            st.subheader("League Rosters")
            st.json(rosters)
