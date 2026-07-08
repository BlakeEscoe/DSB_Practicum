import streamlit as st
from sleeper_connect import (
    get_user_id,
    get_leagues,
    get_all_rosters,
    get_user_name,
    nfl_player_ids,
)
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Fantasy Football Predictor", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_player_lookup():
    players_df = pd.read_csv(
        DATA_DIR / "main_df_with_sleeper_ids.csv",
        usecols=[
            "sleeper_id",
            "player_display_name",
            "position_player_stats",
            "team_player_stats",
        ],
    )
    players_df["sleeper_id"] = (
        pd.to_numeric(players_df["sleeper_id"], errors="coerce")
        .astype("Int64")
        .astype("string")
    )

    return (
        players_df.dropna(subset=["sleeper_id"])
        .drop_duplicates("sleeper_id")
        .set_index("sleeper_id")
        .to_dict("index")
    )


@st.cache_data
def cached_user_name(user_id):
    return get_user_name(user_id)


@st.cache_data
def load_sleeper_player_lookup():
    sleeper_players = nfl_player_ids()

    if isinstance(sleeper_players, str) and sleeper_players.startswith("Error"):
        return {}

    return {
        str(player_id): {
            "player_display_name": player_info.get("full_name", "Unknown"),
            "position_player_stats": player_info.get("position", ""),
            "team_player_stats": player_info.get("team", ""),
        }
        for player_id, player_info in sleeper_players.items()
    }


def build_roster_df(roster):
    player_lookup = load_player_lookup()
    sleeper_player_lookup = load_sleeper_player_lookup()
    starter_ids = {
        str(sleeper_id)
        for sleeper_id in roster.get("starters", [])
        if sleeper_id is not None
    }
    players = []

    for sleeper_id in roster.get("players", []) or []:
        sleeper_id_str = str(sleeper_id)
        lineup_status = "Starter" if sleeper_id_str in starter_ids else "Bench"

        # Handle Team Defense (DST)
        if sleeper_id_str.isalpha():
            players.append(
                {
                    "Lineup": lineup_status,
                    "Player": f"{sleeper_id_str} Defense",
                    "Position": "DEF",
                    "Team": sleeper_id_str,
                    "Sleeper ID": sleeper_id_str,
                }
            )

        # Handle regular players
        else:
            info = player_lookup.get(sleeper_id_str) or sleeper_player_lookup.get(
                sleeper_id_str, {}
            )

            players.append(
                {
                    "Lineup": lineup_status,
                    "Player": info.get("player_display_name", "Unknown"),
                    "Position": info.get("position_player_stats", ""),
                    "Team": info.get("team_player_stats", ""),
                    "Sleeper ID": sleeper_id_str,
                }
            )

    roster_df = pd.DataFrame(players)

    if roster_df.empty:
        return roster_df

    roster_df["Lineup Order"] = roster_df["Lineup"].map({"Starter": 0, "Bench": 1})

    return (
        roster_df.sort_values(["Lineup Order", "Position", "Player"])
        .drop(columns="Lineup Order")
        .reset_index(drop=True)
    )


def show_roster(roster):
    owner = roster.get("owner_id", "Unknown")
    owner_name = cached_user_name(owner) if owner != "Unknown" else "Unknown"
    st.caption(f"Fantasy General Manager: {owner_name}")

    roster_df = build_roster_df(roster)

    if roster_df.empty:
        st.info("No roster players found.")
        return

    starter_df = roster_df[roster_df["Lineup"] == "Starter"].drop(columns="Lineup")
    bench_df = roster_df[roster_df["Lineup"] == "Bench"].drop(columns="Lineup")

    with st.expander(f"Active Roster ({len(starter_df)})", expanded=True):
        st.dataframe(
            starter_df,
            hide_index=True,
            use_container_width=True,
        )

    with st.expander(f"Bench Players ({len(bench_df)})"):
        if bench_df.empty:
            st.caption("No bench players found.")
        else:
            st.dataframe(
                bench_df,
                hide_index=True,
                use_container_width=True,
            )


st.title("Fantasy Pathfinder")

st.markdown(
    """
    <hr style="border: 1px solid #ccc;">
    """,
    unsafe_allow_html=True,
)

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
            st.session_state["rosters"] = rosters

    if "rosters" in st.session_state:
        rosters = st.session_state["rosters"]
        user_id = st.session_state.get("user_id")

        st.subheader("League Rosters")

        user_roster = None
        other_rosters = []

        for roster in rosters:
            if roster.get("owner_id") == user_id:
                user_roster = roster
            else:
                other_rosters.append(roster)

        if user_roster:
            st.markdown("## Your Team")
            show_roster(user_roster)
            st.divider()

        for i, roster in enumerate(other_rosters):
            owner = roster.get("owner_id", "Unknown")
            owner_name = cached_user_name(owner) if owner != "Unknown" else "Unknown"

            with st.expander(f"Team {owner_name}"):
                show_roster(roster)
