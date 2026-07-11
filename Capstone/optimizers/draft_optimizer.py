import pandas as pd


def rank_players(players, drafted_players=None):
    # Functions let us reuse code instead of rewriting the same steps in multiple places.
    # This function can rank players from a CSV file, a web app, or a future test.

    # Create replacement level values for each position.
    replacement_levels = {
        "QB": 17,
        "RB": 10,
        "WR": 10,
        "TE": 7,
    }

    # Make a copy so we do not accidentally change the original DataFrame.
    ranked_players = players.copy()

    # drafted_players is optional.
    # If we receive a list of drafted player names, remove those players before ranking.
    if drafted_players is not None:
        ranked_players = ranked_players[
            ~ranked_players["player"].isin(drafted_players)
        ]

    # Add a replacement_points column by matching each player's position.
    ranked_players["replacement_points"] = ranked_players["position"].map(
        replacement_levels
    )

    # Calculate Value Over Replacement (VOR).
    # VOR shows how many points a player is projected to score above a replacement-level player.
    ranked_players["value_over_replacement"] = (
        ranked_players["predicted_points"] - ranked_players["replacement_points"]
    )

    # Sort players from highest Value Over Replacement to lowest Value Over Replacement.
    ranked_players = ranked_players.sort_values(
        by="value_over_replacement", ascending=False
    )

    # Return the ranked DataFrame so other files can use it.
    return ranked_players


# This block only runs when we run this file directly.
# It will not run if another file imports rank_players().
if __name__ == "__main__":
    players = pd.read_csv("Capstone/data/sample_player_predictions.csv")

    ranked_players = rank_players(players)

    print(ranked_players.to_string(index=False))
