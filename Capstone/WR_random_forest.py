import numpy as np
import pandas as pd

from position_model_engine import (
    run_player_position_model,
)


def add_wr_derived_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["total_yards"] = dataframe["receiving_yards"].fillna(0) + dataframe[
        "rushing_yards"
    ].fillna(0)

    dataframe["total_tds"] = dataframe["receiving_tds"].fillna(0) + dataframe[
        "rushing_tds"
    ].fillna(0)

    dataframe["catch_rate"] = np.where(
        dataframe["targets"].gt(0),
        dataframe["receptions"] / dataframe["targets"],
        np.nan,
    )

    dataframe["yards_per_target"] = np.where(
        dataframe["targets"].gt(0),
        dataframe["receiving_yards"] / dataframe["targets"],
        np.nan,
    )

    dataframe["air_yards_per_target"] = np.where(
        dataframe["targets"].gt(0),
        dataframe["receiving_air_yards"] / dataframe["targets"],
        np.nan,
    )

    dataframe["yards_after_catch_per_reception"] = np.where(
        dataframe["receptions"].gt(0),
        dataframe["receiving_yards_after_catch"] / dataframe["receptions"],
        np.nan,
    )

    return dataframe


WR_CONFIG = {
    "model_name": "wr_random_forest",
    "output_prefix": "wr",
    "position_value": "WR",
    "target": "fantasy_points_ppr",
    "validation_season": 2024,
    "test_season": 2025,
    "minimum_model_season": 2001,
    "derived_builder": add_wr_derived_features,
    "numeric_raw_columns": [
        "fantasy_points_ppr",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "receiving_first_downs",
        "receiving_epa",
        "racr",
        "target_share",
        "air_yards_share",
        "wopr",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "temp_player_stats",
        "wind_player_stats",
        "spread_line_player_stats",
        "total_line_player_stats",
        "away_rest_player_stats",
        "home_rest_player_stats",
        "div_game_player_stats",
    ],
    "categorical_raw_columns": [
        "team_player_stats",
        "opponent_team",
        "away_team_player_stats",
        "home_team_player_stats",
        "roof_player_stats",
        "surface_player_stats",
    ],
    "last_game_columns": [
        "fantasy_points_ppr",
        "targets",
        "receptions",
        "receiving_yards",
        "target_share",
    ],
    "rolling_windows": {
        "fantasy_points_ppr": [3, 5],
        "targets": [3, 5],
        "receptions": [3, 5],
        "receiving_yards": [3, 5],
        "receiving_tds": [5],
        "receiving_air_yards": [3, 5],
        "receiving_yards_after_catch": [3, 5],
        "receiving_first_downs": [5],
        "target_share": [3, 5],
        "air_yards_share": [3, 5],
        "wopr": [3, 5],
        "racr": [5],
        "carries": [5],
        "rushing_yards": [5],
        "rushing_tds": [5],
        "total_yards": [3, 5],
        "total_tds": [5],
        "catch_rate": [5],
        "yards_per_target": [5],
        "air_yards_per_target": [5],
        "yards_after_catch_per_reception": [5],
    },
    "season_average_columns": [
        "fantasy_points_ppr",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "target_share",
        "air_yards_share",
        "wopr",
        "total_yards",
        "total_tds",
    ],
    "career_average_columns": [
        "fantasy_points_ppr",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "target_share",
        "total_yards",
    ],
    "previous_season_columns": [
        "fantasy_points_ppr",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "receiving_air_yards",
        "target_share",
        "air_yards_share",
        "wopr",
        "total_yards",
    ],
    "ewma_columns": [
        "fantasy_points_ppr",
        "targets",
        "receptions",
        "receiving_yards",
        "target_share",
        "wopr",
    ],
    "ewma_span": 3,
    "context_numeric_columns": [
        "season",
        "week",
        "temp_player_stats",
        "wind_player_stats",
        "spread_line_player_stats",
        "total_line_player_stats",
        "away_rest_player_stats",
        "home_rest_player_stats",
        "div_game_player_stats",
    ],
    "context_categorical_columns": [
        "team_player_stats",
        "opponent_team",
        "roof_player_stats",
        "surface_player_stats",
    ],
    "missing_indicator_columns": [
        "temp_player_stats",
        "wind_player_stats",
        "spread_line_player_stats",
        "total_line_player_stats",
    ],
    "betting_columns": [
        "spread_line_player_stats",
        "total_line_player_stats",
    ],
}


if __name__ == "__main__":
    run_player_position_model(WR_CONFIG)
