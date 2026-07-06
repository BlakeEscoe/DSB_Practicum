import streamlit as st
from sleeper_connect import get_user_id, get_leagues, get_all_rosters
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Fantasy Football Predictor", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data"

players_df = pd.read_csv(DATA_DIR / "main_df_with_sleeper_ids.csv")
players_df["sleeper_id"] = (
    pd.to_numeric(players_df["sleeper_id"], errors="coerce")
    .astype("Int64")
    .astype("string")
)


player_lookup = (
    players_df[
        [
            "sleeper_id",
            "player_display_name",
            "position_player_stats",
            "team_player_stats",
        ]
    ]
    .dropna(subset=["sleeper_id"])
    .drop_duplicates("sleeper_id")
    .set_index("sleeper_id")
    .to_dict("index")
)


st.title("Fantasy Football Expected Points Predictor")

st.write("Enter your Sleeper username to find your fantasy leagues and view roster data.")

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

            for i, roster in enumerate(rosters):
                st.markdown(f"## Team {i + 1}")

                owner = roster.get("owner_id", "Unknown")
                st.caption(f"Owner ID: {owner}")

                players = []

                for sleeper_id in roster.get("players", []):
                    sleeper_id_str = str(sleeper_id)

                    # Handle Team Defense (DST)
                    if sleeper_id_str.isalpha():
                        players.append(
                            {
                                "Player": f"{sleeper_id_str} Defense",
                                "Position": "DEF",
                                "Team": sleeper_id_str,
                                "Sleeper ID": sleeper_id_str,
                            }
                        )

                    # Handle regular players
                    else:
                        info = player_lookup.get(sleeper_id_str, {})

                        players.append(
                            {
                                "Player": info.get(
                                    "player_display_name", "Unknown"
                                ),
                                "Position": info.get(
                                    "position_player_stats", ""
                                ),
                                "Team": info.get(
                                    "team_player_stats", ""
                                ),
                                "Sleeper ID": sleeper_id_str,
                            }
                        )

                roster_df = pd.DataFrame(players)

                st.dataframe(
                    roster_df,
                    hide_index=True,
                    use_container_width=True,
                )

                st.divider()