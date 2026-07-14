import numpy as np
import pandas as pd

from position_model_engine import (
    run_player_position_model,
)


def add_rb_derived_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["touches"] = dataframe["carries"].fillna(0) + dataframe[
        "receptions"
    ].fillna(0)

    dataframe["total_yards"] = dataframe["rushing_yards"].fillna(0) + dataframe[
        "receiving_yards"
    ].fillna(0)

    dataframe["total_tds"] = dataframe["rushing_tds"].fillna(0) + dataframe[
        "receiving_tds"
    ].fillna(0)

    dataframe["yards_per_carry"] = np.where(
        dataframe["carries"].gt(0),
        dataframe["rushing_yards"] / dataframe["carries"],
        np.nan,
    )

    dataframe["yards_per_target"] = np.where(
        dataframe["targets"].gt(0),
        dataframe["receiving_yards"] / dataframe["targets"],
        np.nan,
    )

    dataframe["catch_rate"] = np.where(
        dataframe["targets"].gt(0),
        dataframe["receptions"] / dataframe["targets"],
        np.nan,
    )

    return dataframe


RB_CONFIG = {
    "model_name": "rb_random_forest",
    "output_prefix": "rb",
    "position_value": "RB",
    "target": "fantasy_points_ppr",
    "validation_season": 2024,
    "test_season": 2025,
    "minimum_model_season": 2001,
    "derived_builder": add_rb_derived_features,
    "numeric_raw_columns": [
        "fantasy_points_ppr",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_first_downs",
        "rushing_epa",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "receiving_air_yards",
        "receiving_yards_after_catch",
        "receiving_first_downs",
        "receiving_epa",
        "target_share",
        "air_yards_share",
        "wopr",
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
        "carries",
        "targets",
        "touches",
        "total_yards",
    ],
    "rolling_windows": {
        "fantasy_points_ppr": [3, 5],
        "carries": [3, 5],
        "rushing_yards": [3, 5],
        "rushing_tds": [5],
        "rushing_first_downs": [5],
        "targets": [3, 5],
        "receptions": [3, 5],
        "receiving_yards": [3, 5],
        "receiving_tds": [5],
        "receiving_air_yards": [5],
        "receiving_yards_after_catch": [5],
        "target_share": [3, 5],
        "touches": [3, 5],
        "total_yards": [3, 5],
        "total_tds": [5],
        "yards_per_carry": [5],
        "yards_per_target": [5],
        "catch_rate": [5],
    },
    "season_average_columns": [
        "fantasy_points_ppr",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "target_share",
        "touches",
        "total_yards",
        "total_tds",
    ],
    "career_average_columns": [
        "fantasy_points_ppr",
        "carries",
        "rushing_yards",
        "targets",
        "receptions",
        "receiving_yards",
        "touches",
        "total_yards",
    ],
    "previous_season_columns": [
        "fantasy_points_ppr",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "touches",
        "total_yards",
    ],
    "ewma_columns": [
        "fantasy_points_ppr",
        "carries",
        "targets",
        "touches",
        "total_yards",
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
    run_player_position_model(RB_CONFIG)
