import lineups
import pandas as pd
from itertools import combinations


def join_clean(values):
    return ", ".join(
        str(value)
        for value in values
        if pd.notna(value) and str(value).strip()
    )


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

def get_trade_candidates(
    team,
    starting_positions,
    num_worst_starters=4,
    num_best_bench=2
):
    """
    Returns likely trade candidates:
      - Worst projected starters (excluding K)
      - Best projected bench players
    """

    lineup, _ = lineups.optimize_single_roster(
        team,
        starting_positions
    )

    starters = lineup[
        lineup["starting"]
    ].copy()

    bench = lineup[
        ~lineup["starting"]
    ].copy()

    # Ignore kickers
    starters = starters[
        starters["position"] != "K"
    ]

    bench = bench[
        bench["position"] != "K"
    ]

    worst_starters = (
        starters
        .sort_values(
            "projected_points",
            ascending=True
        )
        .head(num_worst_starters)
    )

    best_bench = (
        bench
        .sort_values(
            "projected_points",
            ascending=False
        )
        .head(num_best_bench)
    )

    candidates = pd.concat(
        [worst_starters, best_bench]
    ).drop_duplicates(
        subset="player_id"
    )

    return team[
        team["player_id"].isin(
            candidates["player_id"]
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

    trade_candidates = get_trade_candidates(
        team_a,
        starting_positions,
        num_worst_starters=1,
        num_best_bench=1
    )

    possible_trades = []

    counter = 0

    # Compare team_a against every other team
    for team_b_id, team_b in teams.items():

        if team_b_id == team_a_id:
            continue

        # Try only the weak starters from team_a
        # Build all 1-player and 2-player packages

        team_a_packages = []

        # 1-for-1 candidates
        for _, player in trade_candidates.iterrows():
            team_a_packages.append(
                trade_candidates[
                    trade_candidates["player_id"] == player["player_id"]
                    ]
            )

        # 2-player packages
        for combo in combinations(
                trade_candidates.index,
                2
        ):
            team_a_packages.append(
                trade_candidates.loc[list(combo)]
            )

        team_b_candidates = get_trade_candidates(
            team_b,
            starting_positions,
            num_worst_starters=2,
            num_best_bench=0
        )

        team_b_packages = []

        # 1-player packages
        for _, player in team_b_candidates.iterrows():
            team_b_packages.append(
                team_b_candidates[
                    team_b_candidates["player_id"] == player["player_id"]
                    ]
            )

        # 2-player packages
        for combo in combinations(
                team_b_candidates.index,
                2
        ):
            team_b_packages.append(
                team_b_candidates.loc[list(combo)]
            )

        for players_to_team_b in team_a_packages:

            for players_to_team_a in team_b_packages:

                counter += 1

                result = evaluate_trade(
                    team_a,
                    team_b,
                    players_to_team_a,
                    players_to_team_b,
                    starting_positions
                )

                gain_a = result["team_a"]["weekly_gain"]
                gain_b = result["team_b"]["weekly_gain"]

                if gain_a >= min_gain and gain_b >= min_gain:
                    possible_trades.append({

                        "team_a":
                            team_a.iloc[0]["fantasy_team"],

                        "team_b":
                            team_b.iloc[0]["fantasy_team"],

                        "team_a_gives":
                            join_clean(
                                players_to_team_b["player_name"]
                            ),

                        "team_b_gives":
                            join_clean(
                                players_to_team_a["player_name"]
                            ),

                        "team_a_gain":
                            round(gain_a, 2),

                        "team_b_gain":
                            round(gain_b, 2),

                        "combined_gain":
                            round(
                                gain_a + gain_b,
                                2
                            ),

                        "players_each":
                            len(players_to_team_a)
                    })

    print(counter)

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

def get_best_trades_by_position(league_id, team_id, position, optimized_lineups, scoring_parameters):
    # scoring_parameters can be "past_stats", "projections"

    if scoring_parameters == 'projections':
        optimized_lineups['score_value'] = optimized_lineups['value_over_replacement']
    else:
        optimized_lineups['score_value'] = optimized_lineups['ppr_ppg']

    current_lineups = optimized_lineups[optimized_lineups["position"] != 'K']
    current_lineups = current_lineups[current_lineups["position"] != 'DEF']

    current_lineups["weight"] = current_lineups["score_value"] * (
            1 + 2 * current_lineups["starting"].astype(int)
    )

    team_a_players = current_lineups[
        current_lineups["roster_id"] == team_id
        ]

    target_value = team_a_players[
        (team_a_players["position"] == position) &
        (team_a_players["starting"])
        ]["score_value"].min()

    trade_candidates = team_a_players[
        (team_a_players["position"] != position) |
        (
                (team_a_players["position"] == position) &
                (team_a_players["score_value"] <= target_value)
        )
        ]

    trade_targets = current_lineups[
        (current_lineups["roster_id"] != team_id) &
        (current_lineups["position"] == position) &
        (current_lineups["score_value"] > target_value)
        ][
        ["fantasy_team", "player_name", "position", "starting", "score_value", "weight"]
    ]

    team_a_packages = []

    # 1-player packages
    for idx in trade_candidates.index:
        package = trade_candidates.loc[[idx]]

        team_a_packages.append({
            "fantasy_team": package["fantasy_team"].iloc[0],
            "players": join_clean(package["player_name"]),
            "position": join_clean(package["position"]),
            "starting": package["starting"].iloc[0],
            "score_value": package["score_value"].sum(),
            "weight": package["weight"].sum()
        })

    # 2-player packages
    for combo in combinations(trade_candidates.index, 2):
        package = trade_candidates.loc[list(combo)]

        team_a_packages.append({
            "fantasy_team": join_clean(package["fantasy_team"].unique()),
            "players": join_clean(package["player_name"]),
            "position": join_clean(package["position"]),
            "starting": join_clean(package["starting"].astype(str)),
            "score_value": package["score_value"].sum(),
            "weight": package["weight"].sum()
        })

    team_a_packages = pd.DataFrame(team_a_packages)

    trades = team_a_packages.merge(
        trade_targets,
        how="cross",
        suffixes=("_team_a", "_team_b")
    )

    trades["weight_diff"] = trades["weight_team_a"] - trades["weight_team_b"]

    trades = (
        trades[
            (trades["weight_diff"] >= 0) &
            (trades["score_value_team_a"]/trades["score_value_team_b"] <= 1.15)
            ]
        .sort_values(
            "weight_diff",
            ascending=True
        )
    )

    return trades

def get_trades_to_improve_both_starting_lineups(league_id, team_id, optimized_lineups, scoring_parameters):
    # scoring_parameters can be "past_stats", "projections"

    if scoring_parameters == 'projections':
        optimized_lineups['score_value'] = optimized_lineups['value_over_replacement']
    else:
        optimized_lineups['score_value'] = optimized_lineups['ppr_ppg']

    current_lineups = optimized_lineups[optimized_lineups["position"] != 'K']
    current_lineups = current_lineups[current_lineups["position"] != 'DEF']

    team_a_players = current_lineups[
        current_lineups["roster_id"] == team_id
        ]

    positions = ["QB", "RB", "WR", "TE"]

    trades = []

    for position in positions:
        target_value = team_a_players[
            (team_a_players["position"] == position) &
            (team_a_players["starting"])
            ]["score_value"].min()

        player_to_be_traded = team_a_players[(team_a_players["position"] == position) &
                    (team_a_players["score_value"] == target_value)]

        trade_candidates = team_a_players[
            (team_a_players["position"] != position) |
            (
                    (team_a_players["position"] == position) &
                    (team_a_players["score_value"] < target_value)
            )
            ]

        for team_b_id in current_lineups["roster_id"].unique():
            if team_b_id == team_id:
                continue

            trade_target = (
                current_lineups[
                    (current_lineups["roster_id"] == team_b_id) &
                    (current_lineups["position"] == position) &
                    (current_lineups["score_value"] > target_value)
                    ][
                    ["fantasy_team", "player_name", "position", "starting", "score_value"]
                ]
                .nsmallest(1, "score_value")
            )

            if trade_target.empty:
                continue

            trade_target_name = trade_target["player_name"].iloc[0]

            other_team_b_players = current_lineups[
                (current_lineups["roster_id"] == team_b_id) &
                (current_lineups["starting"]) &
                (current_lineups["player_name"] != trade_target_name)
                ][
                ["fantasy_team", "player_name", "position", "starting", "score_value"]
            ]

            trade_id = 0

            for _, player in other_team_b_players.iterrows():
                position_2 = player["position"]
                target_value_2 = player["score_value"]
                trade_target_packages = []
                trade_id += 1

                package = pd.concat(
                    [
                        trade_target,
                        player.to_frame().T
                    ],
                    ignore_index=True
                )

                trade_target_packages.append({
                    "fantasy team": join_clean(package["fantasy_team"].unique()),
                    "players": join_clean(package["player_name"]),
                    "position": join_clean(package["position"]),
                    "starting": join_clean(package["starting"].astype(str)),
                    "score_value": package["score_value"].sum(),
                    "trade_id": trade_id
                })

                team_b_sent_value = player["score_value"] + trade_target["score_value"].iloc[0]

                team_a_eligible_players = trade_candidates[(trade_candidates["position"] == position_2)
                                                           & (trade_candidates["score_value"] > target_value_2)
                                                           & (~(trade_candidates["starting"]) | (trade_candidates["score_value"] + target_value <= team_b_sent_value))]

                team_a_packages = []

                for _, player_a in team_a_eligible_players.iterrows():
                    package = pd.concat(
                        [
                            player_to_be_traded,
                            player_a.to_frame().T
                        ],
                        ignore_index=True
                    )

                    team_a_packages.append({
                        "fantasy team": join_clean(package["fantasy_team"].unique()),
                        "players": join_clean(package["player_name"]),
                        "position": join_clean(package["position"]),
                        "starting": join_clean(package["starting"].astype(str)),
                        "score_value": package["score_value"].sum(),
                        "trade_id": trade_id
                    })


                team_a_packages = pd.DataFrame(team_a_packages)
                trade_target_packages = pd.DataFrame(trade_target_packages)

                """
                print("")
                print(player_to_be_traded)
                print(trade_target)
                print(team_a_packages)
                print(trade_target_packages)
                """

                if team_a_packages.empty or trade_target_packages.empty:
                    continue

                trades.extend(
                    team_a_packages.merge(
                        trade_target_packages,
                        how="left",
                        on="trade_id",
                        suffixes=("_team_a", "_team_b")
                    ).to_dict("records")
                )

    trades = pd.DataFrame(trades)
    #trades = trades[trades["ppr_ppg_team_a"] >= trades["ppr_ppg_team_b"]]

    return trades


if __name__ == "__main__":
    optimized_lineups = lineups.optimize_starting_lineups(
        '1319148515363921920',
        scoring_parameters="actual_ppg"
    )

    get_best_trades_by_position(league_id='1319148515363921920', team_id=4, position='RB', optimized_lineups=optimized_lineups,scoring_parameters='past_stats').to_csv('RB_trades.csv')
    get_best_trades_by_position(league_id='1319148515363921920', team_id=4, position='WR', optimized_lineups=optimized_lineups,scoring_parameters='projections').to_csv('WR_trades.csv')

    get_trades_to_improve_both_starting_lineups(
        league_id='1319148515363921920',
        team_id=4,
        optimized_lineups=optimized_lineups,
        scoring_parameters="past_stats"
    ).to_csv('trades.csv')
