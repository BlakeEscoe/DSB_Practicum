import lineups
import pandas as pd

def calculate_trade_impact(
    roster_before,
    roster_after,
    starting_positions
):
    """
    Calculates how much a roster improves after a trade.
    """

    _, before_points = lineups.optimize_single_roster(
        roster_before,
        starting_positions
    )

    lineup_after, after_points = lineups.optimize_single_roster(
        roster_after,
        starting_positions
    )

    return {
        "before_points": before_points,
        "after_points": after_points,
        "weekly_gain": after_points - before_points,
        "new_lineup": lineup_after[
            lineup_after["starting"]
        ]
    }


def evaluate_trade(
    team_a_before,
    team_b_before,
    players_to_team_a,
    players_to_team_b,
    starting_positions
):
    """
    Evaluate a trade between two teams.

    players_to_team_a and players_to_team_b
    should be player dataframe rows.
    """

    team_a_after = pd.concat(
        [
            team_a_before,
            players_to_team_a
        ]
    )

    team_a_after = team_a_after[
        ~team_a_after["player_id"].isin(
            players_to_team_b["player_id"]
        )
    ]


    team_b_after = pd.concat(
        [
            team_b_before,
            players_to_team_b
        ]
    )

    team_b_after = team_b_after[
        ~team_b_after["player_id"].isin(
            players_to_team_a["player_id"]
        )
    ]


    return {
        "team_a": calculate_trade_impact(
            team_a_before,
            team_a_after,
            starting_positions
        ),

        "team_b": calculate_trade_impact(
            team_b_before,
            team_b_after,
            starting_positions
        )
    }


def get_worst_starters(team, starting_positions, num_players=3):
    """
    Returns the lowest projected starters on a roster.
    """

    lineup, _ = lineups.optimize_single_roster(
        team,
        starting_positions
    )

    starters = lineup[
        lineup["starting"]
    ].copy()

    # Exclude kickers
    starters = starters[
        starters["position"] != "K"
    ]

    worst_starters = (
        starters
        .sort_values(
            "projected_points",
            ascending=True
        )
        .head(num_players)
    )

    return team[
        team["player_id"].isin(
            worst_starters["player_id"]
        )
    ]

def find_mutually_beneficial_trades(
    league_id,
    team_id,
    min_gain=0.1
):

    rosters = lineups.get_league_rosters(league_id)
    starting_positions = lineups.get_league_starting_positions(league_id)

    # Group teams
    teams = {
        roster_id: team.copy()
        for roster_id, team in rosters.groupby("roster_id")
    }

    if team_id not in teams:
        raise ValueError(
            f"Team ID {team_id} not found in league rosters"
        )

    # Selected team is always team_a
    team_a_id = team_id
    team_a = teams[team_a_id]


    trade_candidates = get_worst_starters(
        team_a,
        starting_positions,
        num_players=2
    )

    lineup, _ = lineups.optimize_single_roster(
        team_a,
        starting_positions
    )

    starters = lineup[
        lineup["starting"]
    ].copy()

    # Exclude kickers
    trade_candidates = starters[
        starters["position"] != "K"
    ]

    possible_trades = []

    # Compare team_a against every other team
    for team_b_id, team_b in teams.items():

        if team_b_id == team_a_id:
            continue

        # Try only the weak starters from team_a
        for _, player_a in trade_candidates.iterrows():

            for _, player_b in team_b.iterrows():

                players_to_team_a = pd.DataFrame(
                    [player_b]
                )

                players_to_team_b = pd.DataFrame(
                    [player_a]
                )

                result = evaluate_trade(
                    team_a,
                    team_b,
                    players_to_team_a,
                    players_to_team_b,
                    starting_positions
                )

                gain_a = result["team_a"]["weekly_gain"]
                gain_b = result["team_b"]["weekly_gain"]

                # Both teams improve
                if (gain_a >= min_gain
                    and gain_b >= min_gain
                ):

                    possible_trades.append({

                        "team_a": team_a.iloc[0]["fantasy_team"],
                        "team_b": team_b.iloc[0]["fantasy_team"],

                        "team_a_gives":
                            player_a["player_name"],

                        "team_b_gives":
                            player_b["player_name"],

                        "team_a_gain":
                            round(gain_a, 2),

                        "team_b_gain":
                            round(gain_b, 2),

                        "combined_gain":
                            round(
                                gain_a + gain_b,
                                2
                            )
                    })


    if not possible_trades:
        return pd.DataFrame()


    return (
        pd.DataFrame(possible_trades)
        .sort_values(
            "combined_gain",
            ascending=False
        )
        .reset_index(drop=True)
    )

trades = find_mutually_beneficial_trades(
    league_id="1319148515363921920",
    team_id=4,          # Sleeper roster_id for your team
    min_gain=0.1        # Minimum weekly gain required for both teams
)

# Save results
trades.to_csv(
    "mutually_beneficial_trades.csv",
    index=False
)