import numpy as np
import pandas as pd

from position_model_engine import (
    run_player_position_model,
)


def add_kicker_derived_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    field_goal_distance_columns = [
        "fg_made_0_19",
        "fg_made_20_29",
        "fg_made_30_39",
        "fg_made_40_49",
        "fg_made_50_59",
        "fg_made_60_",
    ]

    for column in field_goal_distance_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe["fg_made"] = pd.to_numeric(
        dataframe["fg_made"],
        errors="coerce",
    )

    dataframe["pat_made"] = pd.to_numeric(
        dataframe["pat_made"],
        errors="coerce",
    )

    # Standard distance-based kicker scoring:
    # 0–39 yards = 3
    # 40–49 yards = 4
    # 50–59 yards = 5
    # 60+ yards = 6
    distance_based_fg_points = (
        dataframe["fg_made_0_19"].fillna(0) * 3
        + dataframe["fg_made_20_29"].fillna(0) * 3
        + dataframe["fg_made_30_39"].fillna(0) * 3
        + dataframe["fg_made_40_49"].fillna(0) * 4
        + dataframe["fg_made_50_59"].fillna(0) * 5
        + dataframe["fg_made_60_"].fillna(0) * 6
    )

    has_distance_data = dataframe[field_goal_distance_columns].notna().any(axis=1)

    # Older rows may not have distance buckets.
    # Use three points per made field goal as a fallback.
    fallback_fg_points = dataframe["fg_made"].fillna(0) * 3

    dataframe["field_goal_fantasy_points"] = np.where(
        has_distance_data,
        distance_based_fg_points,
        fallback_fg_points,
    )

    # One point per successful extra point.
    dataframe["fantasy_points"] = dataframe["field_goal_fantasy_points"] + dataframe[
        "pat_made"
    ].fillna(0)

    dataframe["fg_accuracy"] = np.where(
        dataframe["fg_att"].gt(0),
        dataframe["fg_made"] / dataframe["fg_att"],
        np.nan,
    )

    dataframe["pat_accuracy"] = np.where(
        dataframe["pat_att"].gt(0),
        dataframe["pat_made"] / dataframe["pat_att"],
        np.nan,
    )

    dataframe["kicking_attempts"] = dataframe["fg_att"].fillna(0) + dataframe[
        "pat_att"
    ].fillna(0)

    dataframe["successful_kicks"] = dataframe["fg_made"].fillna(0) + dataframe[
        "pat_made"
    ].fillna(0)

    dataframe["long_field_goals_made"] = (
        dataframe["fg_made_40_49"].fillna(0)
        + dataframe["fg_made_50_59"].fillna(0)
        + dataframe["fg_made_60_"].fillna(0)
    )

    print("\nGenerated kicker target summary:")
    print(dataframe["fantasy_points"].describe().round(3))

    return dataframe


K_CONFIG = {
    "model_name": "k_random_forest",
    "output_prefix": "k",
    "position_value": "K",
    "target": "fantasy_points",
    "validation_season": 2024,
    "test_season": 2025,
    "minimum_model_season": 2001,
    "derived_builder": add_kicker_derived_features,
    "numeric_raw_columns": [
        "fantasy_points",
        "fg_made",
        "fg_att",
        "fg_missed",
        "fg_blocked",
        "fg_long",
        "fg_pct",
        "fg_made_0_19",
        "fg_made_20_29",
        "fg_made_30_39",
        "fg_made_40_49",
        "fg_made_50_59",
        "fg_made_60_",
        "fg_missed_0_19",
        "fg_missed_20_29",
        "fg_missed_30_39",
        "fg_missed_40_49",
        "fg_missed_50_59",
        "fg_missed_60_",
        "pat_made",
        "pat_att",
        "pat_missed",
        "pat_blocked",
        "pat_pct",
        "gwfg_made",
        "gwfg_att",
        "gwfg_missed",
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
        "fantasy_points",
        "fg_att",
        "fg_made",
        "pat_att",
        "pat_made",
        "kicking_attempts",
    ],
    "rolling_windows": {
        "fantasy_points": [3, 5],
        "fg_att": [3, 5],
        "fg_made": [3, 5],
        "fg_missed": [5],
        "fg_long": [5],
        "pat_att": [3, 5],
        "pat_made": [3, 5],
        "pat_missed": [5],
        "fg_made_30_39": [5],
        "fg_made_40_49": [5],
        "fg_made_50_59": [5],
        "fg_made_60_": [5],
        "fg_accuracy": [5],
        "pat_accuracy": [5],
        "kicking_attempts": [3, 5],
        "successful_kicks": [3, 5],
        "long_field_goals_made": [5],
    },
    "season_average_columns": [
        "fantasy_points",
        "fg_att",
        "fg_made",
        "fg_missed",
        "fg_long",
        "pat_att",
        "pat_made",
        "pat_missed",
        "fg_accuracy",
        "pat_accuracy",
        "kicking_attempts",
        "successful_kicks",
        "long_field_goals_made",
    ],
    "career_average_columns": [
        "fantasy_points",
        "fg_att",
        "fg_made",
        "pat_att",
        "pat_made",
        "kicking_attempts",
        "successful_kicks",
    ],
    "previous_season_columns": [
        "fantasy_points",
        "fg_att",
        "fg_made",
        "fg_missed",
        "pat_att",
        "pat_made",
        "pat_missed",
        "kicking_attempts",
        "successful_kicks",
    ],
    "ewma_columns": [
        "fantasy_points",
        "fg_att",
        "fg_made",
        "pat_att",
        "pat_made",
        "kicking_attempts",
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
    run_player_position_model(K_CONFIG)
