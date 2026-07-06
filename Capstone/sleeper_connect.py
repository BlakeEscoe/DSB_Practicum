import requests
import pandas as pd
from pathlib import Path
import re

"""
def load_main_dataset():
    capstone_dir = Path(__file__).resolve().parent
    data_path = capstone_dir / "data" / "main_df.csv"

    print(data_path)
    print(data_path.exists())

    return pd.read_csv(data_path)"""

CAPSTONE_DIR = Path(__file__).resolve().parent
DATA_DIR = CAPSTONE_DIR / "data"


def load_main_dataset():
    return pd.read_csv(DATA_DIR / "main_df.csv")


sleeper_error_codes = {
    400: "Error 400: Bad Request",
    404: "Error 404: Not Found",
    429: "Error 429: Too Many Requests",
    500: "Error 500: Internal Server Error",
    503: "Error 503: Service Unavailable",
}


def get_user_id(username):
    # this method returns the user_id based on an entered username
    # test using my sleeper username: brulism
    url = f"https://api.sleeper.app/v1/user/{username}"
    response = requests.get(url)
    if response.status_code in sleeper_error_codes:
        return sleeper_error_codes[response.status_code]
    elif response.json() == None:
        return f"Error {response.status_code}: User not found"
    else:
        return response.json()["user_id"]


def get_leagues(user_id, season=2026):
    # since a user can be in multiple leagues,
    # this method returns a list of leagues they're in
    url = f"https://api.sleeper.app/v1/user/{user_id}/leagues/nfl/{season}"
    response = requests.get(url)
    league_info = {}
    if response.status_code in sleeper_error_codes:
        return sleeper_error_codes[response.status_code]
    elif response.json() == None:
        return f"Error {response.status_code}: League not found"
    else:
        for league in response.json():
            league_info[league["name"]] = league["league_id"]
        return league_info


def get_all_rosters(league_id):
    # this method returns the roster of a given league
    url = f"https://api.sleeper.app/v1/league/{league_id}/rosters"
    response = requests.get(url)
    if response.status_code in sleeper_error_codes:
        return sleeper_error_codes[response.status_code]
    elif response.json() == None:
        return f"Error {response.status_code}: Roster not found"
    else:
        return response.json()


def nfl_player_ids():
    url = "https://api.sleeper.app/v1/players/nfl"
    response = requests.get(url)
    if response.status_code in sleeper_error_codes:
        return sleeper_error_codes[response.status_code]
    elif response.json() == None:
        return f"Error {response.status_code}: Player IDs not found"
    else:
        return response.json()


def sort_player_ids():
    ids = nfl_player_ids()
    players = []

    for player_id, player_info in ids.items():
        players.append(
            {
                "player_id": player_id,
                "player_name": player_info.get("full_name", ""),
                "gsis_id": player_info.get("gsis_id", ""),
                "stats_id": player_info.get("stats_id", ""),
                "position": player_info.get("position", ""),
                "team": player_info.get("team", ""),
            }
        )
    return players


def create_composite_key(df, name_col, pos_col, team_col):
    # fix NaN teams
    df[team_col] = df[team_col].fillna("fa").astype(str).str.lower().str.strip()
    df[pos_col] = df[pos_col].fillna("unk").astype(str).str.lower().str.strip()
    # 1. Clean the player name
    cleaned_name = df[name_col].astype(str).str.lower()
    # gets rid of null/missing names
    cleaned_name = df[name_col].fillna("").astype(str).str.lower()
    # Strip suffixes (jr, iii, ii, sr) surrounded by spaces or at the end
    cleaned_name = cleaned_name.apply(lambda x: re.sub(r"\b(jr|iii|ii|sr)\b", "", x))
    # Strip all punctuation and whitespace
    cleaned_name = cleaned_name.apply(lambda x: re.sub(r"[^a-z]", "", x))
    # 2. Clean position and team
    pos = df[pos_col].astype(str).str.lower().str.strip()
    team = df[team_col].astype(str).str.lower().str.strip()
    # 3. Combine into a unique composite key
    return cleaned_name + "_" + pos + "_" + team


"""code below is to test the shit"""
sleeper_ids = pd.DataFrame(sort_player_ids())
sleeper_ids = sleeper_ids.rename(columns={"player_id": "sleeper_id"})
main_df = load_main_dataset()

# Make copies so you don't accidentally mess up originals
main = main_df.copy()
sleeper = sleeper_ids.copy()

# CLEAN THE MAIN DATASET
main.dropna(subset=["player_id"], inplace=True)  # drop the 530 nulls
# main.drop_duplicates(subset=['player_id'], inplace=True) #drop the 507692 duplicates, leaving 11025 unique players
main["gsis_id"] = main["gsis_id"].astype("string").str.strip()
sleeper["gsis_id"] = sleeper["gsis_id"].astype("string").str.strip()

main["gsis_id"] = main["gsis_id"].replace("", pd.NA)
sleeper["gsis_id"] = sleeper["gsis_id"].replace("", pd.NA)

main_players = (
    main[
        [
            "player_id",
            "team_player_stats",
            "player_name",
            "player_display_name",
            "position_player_stats",
            "gsis_id",
        ]
    ]
    .dropna(subset=["player_id"])
    .drop_duplicates(subset=["player_id"])
    .copy()
)

sleeper_clean = sleeper[sleeper["player_name"] != "Duplicate Player"]

main_players["composite_key"] = create_composite_key(
    main_players, "player_display_name", "position_player_stats", "team_player_stats"
)
sleeper_clean["composite_key"] = create_composite_key(
    sleeper_clean, "player_name", "position", "team"
)

main_players.dropna(subset=["composite_key"], inplace=True)


# Join Sleeper IDs onto the main stats dataset
merged_df = main_players.merge(
    sleeper_clean[["sleeper_id", "player_name", "position", "team", "composite_key"]],
    on="composite_key",
    how="left",
    suffixes=("_main", "_sleeper"),
)
##################

# --- STEP 1: Build the 'loose_key' on the raw dataframes first ---
# This slices off the team portion (e.g. 'joshallen_qb_buf' becomes 'joshallen_qb')
main_players["loose_key"] = main_players["composite_key"].apply(
    lambda x: "_".join(x.split("_")[:-1]) if pd.notna(x) else x
)
sleeper_clean["loose_key"] = sleeper_clean["composite_key"].apply(
    lambda x: "_".join(x.split("_")[:-1]) if pd.notna(x) else x
)

# --- STEP 2: Re-run your strict merge to get a clean baseline dataframe ---
merged_df = main_players.merge(
    sleeper_clean[["composite_key", "sleeper_id"]], on="composite_key", how="left"
)

# --- STEP 3: Separate the successes from the failures ---
# Those that matched successfully via strict key
matched_strict = merged_df[merged_df["sleeper_id"].notna()]

# Those that failed (sleeper_id is NaN)
failed_strict = merged_df[merged_df["sleeper_id"].isna()].drop(columns=["sleeper_id"])

# --- STEP 4: Merge the failures on the 'loose_key' ---
# Pulling 'player_id' out of sleeper_clean and renaming it right away
matched_loose = failed_strict.merge(
    sleeper_clean[["loose_key", "sleeper_id"]], on="loose_key", how="left"
)

# --- STEP 5: Concatenate them back into a single final dataframe ---
final_df = pd.concat([matched_strict, matched_loose], ignore_index=True)

# --- STEP 6: Drop the temporary tracking key and view the new metrics ---
if "loose_key" in final_df.columns:
    final_df = final_df.drop(columns=["loose_key"])

total = len(final_df)
matched = final_df["sleeper_id"].notna().sum()

# final_df.to_csv("data/main_players_with_sleeper_ids.csv", index=False)

# 1. Strip the temporary 'composite_key' if it exists in final_df to keep it clean
if "composite_key" in final_df.columns:
    final_df_clean = final_df[
        ["player_name", "position_player_stats", "team_player_stats", "sleeper_id"]
    ].drop_duplicates()
else:
    # Use whatever unique columns define a player row in your final_df
    final_df_clean = final_df[
        ["player_name", "position_player_stats", "team_player_stats", "sleeper_id"]
    ].drop_duplicates()

# 2. Merge the sleeper_id back into your main stats dataframe
# Change 'player_name', 'position', 'team' to match the exact column names in your main_df
main_df = main_df.merge(
    final_df_clean,
    on=["player_name", "position_player_stats", "team_player_stats"],
    how="left",
)

output_path = DATA_DIR / "main_df_with_sleeper_ids.csv"
main_df.to_csv(output_path, index=False)

"""
what do we need for sleeper?
1. username (we can get their user_id and, in turn, their league id from this.
    they can also just copy paste the url from their league in 
    but for UX purposes maybe user id is easier)
    
    method: get_user_id(username) -> returns user_id

2. get_leagues, this is done and returns a list of leagues they're in

3. its only 2 methods for now, but to actually use it we 
    gotta do it in the actual app (chat says we should do a streamlit app)

"""
