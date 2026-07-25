import sys
from pathlib import Path

import pandas as pd


pd.set_option("future.infer_string", False)

CAPSTONE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = CAPSTONE_DIR / "data"
SAMPLE_PREDICTIONS_FILE = DATA_DIR / "sample_player_predictions.csv"
COMBINED_MODEL_PREDICTIONS_FILE = DATA_DIR / "all_2025_predictions.csv"

if str(CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(CAPSTONE_DIR))


def _read_prediction_csv(csv_path):
    # Some local Python/pandas builds can crash when Streamlit triggers
    # Arrow-backed string columns. These dtypes keep the CSV read simple.
    return pd.read_csv(
        csv_path,
        dtype={
            "player_name": object,
            "position": object,
            "season": int,
            "week": int,
            "predicted_fantasy_points": float,
        },
    )


def _normalize_player_name(player_name):
    # Sleeper and model data can use slightly different punctuation.
    # Normalizing lets "Amon-Ra St. Brown" match "Amon Ra St Brown".
    return "".join(
        character.lower()
        for character in str(player_name)
        if character.isalnum()
    )


def _prepare_predictions_for_optimizer(predictions):
    # The model output is weekly. The draft optimizer needs one row per player,
    # so we keep both the weekly average and the full-season total.
    players = pd.DataFrame()
    players["player"] = predictions["player_name"].astype(object)
    players["position"] = (
        predictions["position"]
        .astype(object)
        .map(lambda position: str(position).upper().strip())
    )
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
        .groupby(
            ["player", "sleeper_id", "position", "team"],
            dropna=False,
            as_index=False,
        )
        .agg(
            predicted_points=("predicted_points", "mean"),
            season_predicted_points=("predicted_points", "sum"),
        )
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
            predictions = _read_prediction_csv(COMBINED_MODEL_PREDICTIONS_FILE)
            players = _prepare_predictions_for_optimizer(predictions)

            return (
                players,
                False,
                COMBINED_MODEL_PREDICTIONS_FILE.name,
            )

        sample_players = pd.read_csv(SAMPLE_PREDICTIONS_FILE)
        sample_players["season_predicted_points"] = (
            sample_players["predicted_points"] * 17
        )

        return (
            sample_players,
            True,
            f"sample fallback ({model_error})",
        )


def rank_players(players, drafted_players=None, drafted_sleeper_ids=None):
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

    # Older/sample inputs may not have a season total, so create one if needed.
    if "season_predicted_points" not in ranked_players.columns:
        ranked_players["season_predicted_points"] = (
            ranked_players["predicted_points"] * 17
        )

    # drafted_sleeper_ids is optional.
    # If Sleeper IDs are available, they are the safest way to remove drafted players.
    if drafted_sleeper_ids is not None and "sleeper_id" in ranked_players.columns:
        drafted_sleeper_ids = {
            str(sleeper_id)
            for sleeper_id in drafted_sleeper_ids
            if sleeper_id is not None
        }
        ranked_players = ranked_players[
            ~ranked_players["sleeper_id"].astype(str).isin(drafted_sleeper_ids)
        ]

    # drafted_players is optional.
    # If we receive player names, remove matching names before ranking.
    # The normalized version handles small punctuation differences.
    if drafted_players is not None:
        drafted_player_names = {
            _normalize_player_name(player)
            for player in drafted_players
            if player
        }
        ranked_players = ranked_players[
            ~ranked_players["player"].map(_normalize_player_name).isin(drafted_player_names)
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
