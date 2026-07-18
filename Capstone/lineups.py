import sleeper_connect
import requests
import pandas as pd
import nflreadpy as nfl
from collections import Counter

def get_league_users(league_id):
    url = f"https://api.sleeper.app/v1/league/{league_id}/users"
    response = requests.get(url)

    sleeper_error_codes = {
        400: "Error 400: Bad Request",
        404: "Error 404: Not Found",
        429: "Error 429: Too Many Requests",
        500: "Error 500: Internal Server Error",
        503: "Error 503: Service Unavailable",
    }

    if response.status_code in sleeper_error_codes:
        return sleeper_error_codes[response.status_code]
    elif response.json() is None:
        return f"Error {response.status_code}: Users not found"
    else:
        return response.json()


def get_player_stats(season=2025):
    player_stats = nfl.load_player_stats(list(range(2000, 2026))).to_pandas()
    player_stats = player_stats[(player_stats['season'] == season) & (player_stats['season_type'] == 'REG')]
    player_stats['fantasy_ppr_points'] = player_stats['passing_tds']*4 + player_stats['passing_yards']*0.04 + player_stats['passing_interceptions']*-1 + player_stats['passing_2pt_conversions']*2 + player_stats['rushing_yards']*0.1 + player_stats['receiving_yards']*0.1 + player_stats['rushing_tds']*6 + player_stats['receiving_tds']*6 + player_stats['rushing_2pt_conversions']*2 + player_stats['receiving_2pt_conversions']*2 + player_stats['receptions']*1 + player_stats['fumbles_lost_total']*-1
    return player_stats.groupby('player_display_name').agg(ppr_ppg=('fantasy_ppr_points', 'mean'))


def get_league_starting_positions(league_id):
    league = requests.get(
        f"https://api.sleeper.app/v1/league/{league_id}"
    ).json()
    positions = league["roster_positions"]
    position_counts = Counter(positions)

    starting_positions_df = pd.DataFrame(
        position_counts.items(),
        columns=["Position", "Starting_Count"]
    ).sort_values("Position").reset_index(drop=True)

    return starting_positions_df

def get_league_rosters(league_id):
    rosters = sleeper_connect.get_all_rosters(league_id)
    stats = get_player_stats(2025)

    # Player metadata
    players = sleeper_connect.nfl_player_ids()

    users = get_league_users(league_id)

    owner_lookup = {
        user["user_id"]: {
            "owner_name": user.get("display_name"),
            "team_name": (
                user.get("metadata", {}).get("team_name")
                or user.get("display_name")
            )
        }
        for user in users
    }

    league_rosters = []

    for roster in rosters:
        owner = owner_lookup.get(roster.get("owner_id"), {})

        for player_id in roster.get("players", []):

            player = players.get(player_id)
            if player is None:
                continue

            league_rosters.append({
                "fantasy_team": owner.get("team_name"),
                "owner_name": owner.get("owner_name"),
                "roster_id": roster["roster_id"],
                "owner_id": roster.get("owner_id"),
                "player_id": player_id,
                "player_name": player.get("full_name"),
                "nfl_team": player.get("team"),
                "position": player.get("position")
            })

    league_rosters = pd.DataFrame(league_rosters)

    return league_rosters.merge(stats, left_on=["player_name"], right_on=["player_display_name"], how="left")

def optimize_single_roster(team, starting_positions):
    """
    Optimizes a single fantasy team's lineup.
    Returns lineup and projected points.
    """

    position_requirements = dict(
        zip(
            starting_positions["Position"],
            starting_positions["Starting_Count"]
        )
    )

    flex_positions = {"FLEX", "SUPER_FLEX"}

    remaining_players = team.sort_values(
        "ppr_ppg",
        ascending=False
    ).copy()

    starters = []

    # Fill normal positions
    for position, count in position_requirements.items():

        if position in flex_positions:
            continue

        eligible = remaining_players[
            remaining_players["position"] == position
        ]

        selected = eligible.head(count).copy()

        if not selected.empty:
            selected["starting_position"] = position
            selected["starting"] = True

            starters.append(selected)

            remaining_players = remaining_players.drop(
                selected.index
            )

    # Fill FLEX/SUPER FLEX
    for slot, count in position_requirements.items():

        if slot not in flex_positions:
            continue

        if slot == "SUPER_FLEX":
            eligible_positions = [
                "QB",
                "RB",
                "WR",
                "TE"
            ]
        else:
            eligible_positions = [
                "RB",
                "WR",
                "TE"
            ]

        eligible = remaining_players[
            remaining_players["position"].isin(
                eligible_positions
            )
        ]

        selected = eligible.head(count).copy()

        if not selected.empty:

            selected["starting_position"] = slot
            selected["starting"] = True

            starters.append(selected)

            remaining_players = remaining_players.drop(
                selected.index
            )

    if starters:
        starters_df = pd.concat(
            starters,
            ignore_index=True
        )
    else:
        starters_df = pd.DataFrame()

    bench = remaining_players.copy()

    bench["starting_position"] = "BENCH"
    bench["starting"] = False

    lineup = pd.concat(
        [
            starters_df,
            bench
        ],
        ignore_index=True
    )

    lineup["projected_points"] = lineup.apply(
        lambda x: x["ppr_ppg"] if x["starting"] else 0,
        axis=1
    )

    return (
        lineup,
        lineup["projected_points"].sum()
    )


def optimize_starting_lineups(league_id, scoring_parameters):
    # scoring_parameters can be "actual_ppg", "projections", "draft_value"

    rosters = get_league_rosters(league_id)
    starting_positions = get_league_starting_positions(league_id)

    results = []

    for roster_id, team in rosters.groupby("roster_id"):

        lineup, points = optimize_single_roster(
            team,
            starting_positions
        )

        lineup["optimal_points"] = points
        lineup["roster_id"] = roster_id

        results.append(lineup)

    return pd.concat(
        results,
        ignore_index=True
    )


rosters = get_league_rosters('1319148515363921920')
rosters.to_csv('rosters.csv')
print(get_league_starting_positions('1319148515363921920'))
optimize_starting_lineups('1319148515363921920',scoring_parameters="actual_ppg").to_csv('optimized_starting_lineups.csv')