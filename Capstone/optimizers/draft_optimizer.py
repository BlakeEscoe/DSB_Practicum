import sys
from pathlib import Path

import pandas as pd


CAPSTONE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CAPSTONE_DIR / "data"
SAMPLE_PREDICTIONS_FILE = DATA_DIR / "sample_player_predictions.csv"
COMBINED_MODEL_PREDICTIONS_FILE = DATA_DIR / "all_2025_predictions.csv"

if str(CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPSTONE_DIR))


def _prepare_predictions_for_optimizer(predictions):
    # The model output is weekly. The draft optimizer needs one row per player,
    # so we average the weekly predictions into one predicted_points value.
    players = pd.DataFrame()
    players["player"] = predictions["player_name"]
    players["position"] = predictions["position"].astype(str).str.upper().str.strip()
    players["predicted_points"] = pd.to_numeric(
        predictions["predicted_fantasy_points"],
        errors="coerce",
    )

    # The current combined prediction file does not include NFL team or Sleeper ID.
    # Keep the columns available so the UI table shape stays consistent.
    players["team"] = pd.NA
    players["sleeper_id"] = pd.NA

    defense_mask = players["position"] == "DEF"
    players.loc[defense_mask, "team"] = players.loc[defense_mask, "player"]

    return (
        players.dropna(subset=["player", "position", "predicted_points"])
        .groupby(["player", "sleeper_id", "position", "team"], dropna=False, as_index=False)
        ["predicted_points"]
        .mean()
    )


def load_player_predictions(season=2025):
    # First try the function your teammates built.
    try:
        from prediction_func import load_predictions

        predictions = load_predictions(season)
        players = _prepare_predictions_for_optimizer(predictions)

        return players, False, f"prediction_func.load_predictions({season})"
    except Exception as model_error:
        # If the six source files are not present but the already-combined model
        # output exists, use that before falling back to the old sample data.
        if COMBINED_MODEL_PREDICTIONS_FILE.exists():
            predictions = pd.read_csv(COMBINED_MODEL_PREDICTIONS_FILE)
            players = _prepare_predictions_for_optimizer(predictions)

            return (
                players,
                False,
                COMBINED_MODEL_PREDICTIONS_FILE.name,
            )

        sample_players = pd.read_csv(SAMPLE_PREDICTIONS_FILE)

        return (
            sample_players,
            True,
            f"sample fallback ({model_error})",
        )


def rank_players(players, drafted_players=None):
    # Functions let us reuse code instead of rewriting the same steps in multiple places.
    # This function can rank players from a CSV file, a web app, or a future test.

    # Create replacement level values for each position.
    replacement_levels = {
        "QB": 17,
        "RB": 10,
        "WR": 10,
        "TE": 7,
        "K": 7,
        "DEF": 6,
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
    players, using_sample_fallback, source = load_player_predictions()

    print(f"Prediction source: {source}")
    if using_sample_fallback:
        print("Using sample fallback projections.")

    ranked_players = rank_players(players)

    print(ranked_players.to_string(index=False))
