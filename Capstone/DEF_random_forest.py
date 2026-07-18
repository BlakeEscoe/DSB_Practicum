from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from position_model_engine import (
    DATA_DIR,
    build_random_forest_pipeline,
    calculate_metrics,
    safe_to_csv,
)


INPUT_FILE = DATA_DIR / "main_df_with_sleeper_ids.csv"

MODEL_DATA_FILE = DATA_DIR / "def_model_dataset.csv"

RESULTS_FILE = DATA_DIR / "def_random_forest_results.csv"

PREDICTIONS_FILE = DATA_DIR / "def_2025_predictions.csv"

MODEL_FILE = DATA_DIR / "def_random_forest_model.joblib"

TARGET = "def_fantasy_points"

VALIDATION_SEASON = 2024
TEST_SEASON = 2025

MIN_BETTING_MAE_IMPROVEMENT = 0.02


# ============================================================
# D/ST SCORING
# Change these values if your league is different.
# ============================================================

SACK_POINTS = 1.0
INTERCEPTION_POINTS = 2.0
FUMBLE_RECOVERY_POINTS = 2.0
SAFETY_POINTS = 2.0
DEFENSIVE_TD_POINTS = 6.0
SPECIAL_TEAMS_TD_POINTS = 6.0
BLOCKED_KICK_POINTS = 2.0


def combine_columns(
    dataframe: pd.DataFrame,
    possible_columns: list[str],
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=dataframe.index,
        dtype="object",
    )

    for column in possible_columns:
        if column in dataframe.columns:
            result = result.combine_first(dataframe[column])

    return result


def first_non_null(
    values: pd.Series,
):
    non_null_values = values.dropna()

    if non_null_values.empty:
        return np.nan

    return non_null_values.iloc[0]


def points_allowed_score(
    points_allowed: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                points_allowed.eq(0),
                points_allowed.between(1, 6),
                points_allowed.between(7, 13),
                points_allowed.between(14, 20),
                points_allowed.between(21, 27),
                points_allowed.between(28, 34),
                points_allowed.ge(35),
            ],
            [
                10.0,
                7.0,
                4.0,
                1.0,
                0.0,
                -1.0,
                -4.0,
            ],
            default=np.nan,
        ),
        index=points_allowed.index,
    )


if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Could not find:\n{INPUT_FILE}")

print(f"Loading: {INPUT_FILE}")

main_df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)


# ============================================================
# CANONICAL GAME AND SCHEDULE COLUMNS
# ============================================================

main_df["def_game_id"] = combine_columns(
    main_df,
    [
        "game_id_player_stats",
        "game_id",
        "game_id_home_team",
    ],
)

main_df["def_away_team"] = combine_columns(
    main_df,
    [
        "away_team_player_stats",
        "away_team_away_team",
    ],
)

main_df["def_home_team"] = combine_columns(
    main_df,
    [
        "home_team_player_stats",
        "home_team_away_team",
    ],
)

main_df["def_away_score"] = combine_columns(
    main_df,
    [
        "away_score_player_stats",
        "away_score_away_team",
    ],
)

main_df["def_home_score"] = combine_columns(
    main_df,
    [
        "home_score_player_stats",
        "home_score_away_team",
    ],
)

main_df["def_roof"] = combine_columns(
    main_df,
    [
        "roof_player_stats",
        "roof_away_team",
    ],
)

main_df["def_surface"] = combine_columns(
    main_df,
    [
        "surface_player_stats",
        "surface_away_team",
    ],
)

main_df["def_temp"] = combine_columns(
    main_df,
    [
        "temp_player_stats",
        "temp_away_team",
    ],
)

main_df["def_wind"] = combine_columns(
    main_df,
    [
        "wind_player_stats",
        "wind_away_team",
    ],
)

main_df["def_spread"] = combine_columns(
    main_df,
    [
        "spread_line_player_stats",
        "spread_line_away_team",
    ],
)

main_df["def_total_line"] = combine_columns(
    main_df,
    [
        "total_line_player_stats",
        "total_line_away_team",
    ],
)

main_df["def_away_rest"] = combine_columns(
    main_df,
    [
        "away_rest_player_stats",
        "away_rest_away_team",
    ],
)

main_df["def_home_rest"] = combine_columns(
    main_df,
    [
        "home_rest_player_stats",
        "home_rest_away_team",
    ],
)

main_df["def_div_game"] = combine_columns(
    main_df,
    [
        "div_game_player_stats",
        "div_game_away_team",
    ],
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "season",
    "week",
    "def_away_score",
    "def_home_score",
    "def_temp",
    "def_wind",
    "def_spread",
    "def_total_line",
    "def_away_rest",
    "def_home_rest",
    "def_div_game",
]

for column in numeric_columns:
    main_df[column] = pd.to_numeric(
        main_df[column],
        errors="coerce",
    )

main_df = main_df.dropna(
    subset=[
        "season",
        "week",
        "def_away_team",
        "def_home_team",
    ]
).copy()

main_df["season"] = main_df["season"].astype(int)

main_df["week"] = main_df["week"].astype(int)


# ============================================================
# CREATE FALLBACK GAME IDS
# ============================================================

invalid_game_id = main_df["def_game_id"].isna() | main_df["def_game_id"].astype(
    str
).str.strip().isin(
    [
        "",
        "nan",
        "None",
    ]
)

main_df.loc[
    invalid_game_id,
    "def_game_id",
] = (
    main_df.loc[
        invalid_game_id,
        "season",
    ].astype(str)
    + "_"
    + main_df.loc[
        invalid_game_id,
        "week",
    ]
    .astype(int)
    .astype(str)
    .str.zfill(2)
    + "_"
    + main_df.loc[
        invalid_game_id,
        "def_away_team",
    ].astype(str)
    + "_"
    + main_df.loc[
        invalid_game_id,
        "def_home_team",
    ].astype(str)
)

main_df["def_game_id"] = main_df["def_game_id"].astype(str)


# ============================================================
# BUILD ONE SCHEDULE ROW PER GAME
# ============================================================

schedule_columns = [
    "def_game_id",
    "season",
    "week",
    "def_away_team",
    "def_home_team",
    "def_away_score",
    "def_home_score",
    "def_roof",
    "def_surface",
    "def_temp",
    "def_wind",
    "def_spread",
    "def_total_line",
    "def_away_rest",
    "def_home_rest",
    "def_div_game",
]

games_df = (
    main_df[schedule_columns]
    .groupby(
        "def_game_id",
        as_index=False,
    )
    .agg(
        {
            column: first_non_null
            for column in schedule_columns
            if column != "def_game_id"
        }
    )
)

games_df = games_df.dropna(
    subset=[
        "def_away_team",
        "def_home_team",
    ]
).copy()


# ============================================================
# CREATE TWO DEFENSE ROWS PER GAME
# ============================================================

home_defense_df = pd.DataFrame(
    {
        "def_game_id": games_df["def_game_id"],
        "season": games_df["season"],
        "week": games_df["week"],
        "team_player_stats": games_df["def_home_team"],
        "opponent_team": games_df["def_away_team"],
        "is_home": 1,
        "points_allowed": games_df["def_away_score"],
        "team_points": games_df["def_home_score"],
        "team_rest": games_df["def_home_rest"],
        "roof_player_stats": games_df["def_roof"],
        "surface_player_stats": games_df["def_surface"],
        "temp_player_stats": games_df["def_temp"],
        "wind_player_stats": games_df["def_wind"],
        "spread_line_player_stats": games_df["def_spread"],
        "total_line_player_stats": games_df["def_total_line"],
        "div_game_player_stats": games_df["def_div_game"],
    }
)

away_defense_df = pd.DataFrame(
    {
        "def_game_id": games_df["def_game_id"],
        "season": games_df["season"],
        "week": games_df["week"],
        "team_player_stats": games_df["def_away_team"],
        "opponent_team": games_df["def_home_team"],
        "is_home": 0,
        "points_allowed": games_df["def_home_score"],
        "team_points": games_df["def_away_score"],
        "team_rest": games_df["def_away_rest"],
        "roof_player_stats": games_df["def_roof"],
        "surface_player_stats": games_df["def_surface"],
        "temp_player_stats": games_df["def_temp"],
        "wind_player_stats": games_df["def_wind"],
        "spread_line_player_stats": games_df["def_spread"],
        "total_line_player_stats": games_df["def_total_line"],
        "div_game_player_stats": games_df["def_div_game"],
    }
)

def_df = pd.concat(
    [
        home_defense_df,
        away_defense_df,
    ],
    ignore_index=True,
)


# ============================================================
# AGGREGATE DEFENSIVE PLAYER STATISTICS
# ============================================================

defensive_stat_columns = [
    "def_sacks",
    "def_interceptions",
    "def_tds",
    "def_safeties",
    "def_qb_hits",
    "def_pass_defended",
    "def_tackles_for_loss",
    "fumble_recovery_opp",
    "special_teams_tds",
]

for column in defensive_stat_columns:
    if column not in main_df.columns:
        main_df[column] = 0

    main_df[column] = pd.to_numeric(
        main_df[column],
        errors="coerce",
    ).fillna(0)

# ============================================================
# CHECK FOR DUPLICATE PLAYER-TEAM-GAME ROWS
# ============================================================

possible_player_identifier = (
    "player_id" if "player_id" in main_df.columns else "player_name"
)

duplicate_player_game_rows = main_df.duplicated(
    subset=[
        "def_game_id",
        "team_player_stats",
        possible_player_identifier,
    ],
    keep=False,
)

print(
    "\nDuplicate player-team-game rows:",
    duplicate_player_game_rows.sum(),
)

if duplicate_player_game_rows.any():
    duplicate_examples = (
        main_df.loc[
            duplicate_player_game_rows,
            [
                "def_game_id",
                "team_player_stats",
                possible_player_identifier,
                "def_sacks",
                "def_interceptions",
                "def_tds",
                "special_teams_tds",
            ],
        ]
        .sort_values(
            [
                "def_game_id",
                "team_player_stats",
                possible_player_identifier,
            ]
        )
        .head(30)
    )

    print("\nExample duplicate rows:")

    print(duplicate_examples.to_string(index=False))


# Keep only one row per player, team, and game before summing
defensive_player_rows = main_df[
    [
        "def_game_id",
        "team_player_stats",
        possible_player_identifier,
        *defensive_stat_columns,
    ]
].drop_duplicates(
    subset=[
        "def_game_id",
        "team_player_stats",
        possible_player_identifier,
    ],
    keep="first",
)


# Aggregate the deduplicated player statistics to team-game level
defensive_stats_df = defensive_player_rows.groupby(
    [
        "def_game_id",
        "team_player_stats",
    ],
    as_index=False,
)[defensive_stat_columns].sum()

def_df = def_df.merge(
    defensive_stats_df,
    on=[
        "def_game_id",
        "team_player_stats",
    ],
    how="left",
    validate="1:1",
)

def_df[defensive_stat_columns] = def_df[defensive_stat_columns].fillna(0)


# ============================================================
# BLOCKED KICKS BELONG TO THE OPPOSING DEFENSE
# ============================================================

for column in [
    "fg_blocked",
    "pat_blocked",
]:
    if column not in main_df.columns:
        main_df[column] = 0

    main_df[column] = pd.to_numeric(
        main_df[column],
        errors="coerce",
    ).fillna(0)

main_df["blocked_attempts"] = main_df["fg_blocked"] + main_df["pat_blocked"]

blocked_against_offense_df = (
    main_df.groupby(
        [
            "def_game_id",
            "team_player_stats",
        ],
        as_index=False,
    )["blocked_attempts"]
    .sum()
    .rename(
        columns={
            "team_player_stats": "opponent_team",
            "blocked_attempts": "blocked_kicks",
        }
    )
)

def_df = def_df.merge(
    blocked_against_offense_df,
    on=[
        "def_game_id",
        "opponent_team",
    ],
    how="left",
    validate="1:1",
)

def_df["blocked_kicks"] = def_df["blocked_kicks"].fillna(0)


# ============================================================
# REMOVE GAMES WITHOUT FINAL SCORES
# ============================================================

missing_score_rows = def_df["points_allowed"].isna().sum()

print(f"\nDefense rows with missing scores: {missing_score_rows:,}")

def_df = def_df.dropna(
    subset=[
        "points_allowed",
    ]
).copy()


# ============================================================
# CREATE D/ST TARGET
# ============================================================

def_df["points_allowed_fantasy"] = points_allowed_score(def_df["points_allowed"])

def_df[TARGET] = (
    def_df["def_sacks"] * SACK_POINTS
    + def_df["def_interceptions"] * INTERCEPTION_POINTS
    + def_df["fumble_recovery_opp"] * FUMBLE_RECOVERY_POINTS
    + def_df["def_safeties"] * SAFETY_POINTS
    + def_df["def_tds"] * DEFENSIVE_TD_POINTS
    + def_df["special_teams_tds"] * SPECIAL_TEAMS_TD_POINTS
    + def_df["blocked_kicks"] * BLOCKED_KICK_POINTS
    + def_df["points_allowed_fantasy"]
)

def_df = def_df.dropna(
    subset=[
        TARGET,
    ]
).copy()

# ============================================================
# REMOVE CLEARLY INVALID HISTORICAL DEFENSIVE STAT ROWS
# ============================================================

invalid_defense_rows = (
    def_df["def_tds"].gt(3)
    | def_df["def_interceptions"].gt(7)
    | def_df["def_sacks"].gt(12)
    | def_df["fumble_recovery_opp"].gt(6)
    | def_df["special_teams_tds"].gt(2)
)

print(
    "\nClearly invalid defense rows removed:",
    invalid_defense_rows.sum(),
)

print("\nInvalid rows by season:")
print(def_df.loc[invalid_defense_rows].groupby("season").size().to_string())

def_df = def_df.loc[~invalid_defense_rows].copy()

# ============================================================
# INSPECT THE HIGHEST D/ST FANTASY SCORES
# ============================================================

defense_target_components = [
    "def_game_id",
    "season",
    "week",
    "team_player_stats",
    "opponent_team",
    "points_allowed",
    "points_allowed_fantasy",
    "def_sacks",
    "def_interceptions",
    "fumble_recovery_opp",
    "def_safeties",
    "def_tds",
    "special_teams_tds",
    "blocked_kicks",
    TARGET,
]

print("\nHighest D/ST fantasy scores:")

print(
    def_df[defense_target_components]
    .sort_values(
        TARGET,
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)

print("\nGenerated defense target summary:")
print(def_df[TARGET].describe().round(3))

print("\nTeam-defense rows by season:")
print(def_df.groupby("season").size().tail(10))


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

def_df = def_df.sort_values(
    [
        "team_player_stats",
        "season",
        "week",
        "def_game_id",
    ],
    kind="mergesort",
).copy()

# ============================================================
# REMOVE DUPLICATE TEAM-WEEK DEFENSE ROWS
# ============================================================

rows_before_deduplication = len(def_df)

duplicate_defense_rows = def_df.duplicated(
    subset=[
        "team_player_stats",
        "season",
        "week",
    ],
    keep=False,
)

print(
    "\nDuplicate defense team-week rows before removal:",
    duplicate_defense_rows.sum(),
)

if duplicate_defense_rows.any():
    duplicate_examples = (
        def_df.loc[
            duplicate_defense_rows,
            [
                "def_game_id",
                "season",
                "week",
                "team_player_stats",
                "opponent_team",
                TARGET,
            ],
        ]
        .sort_values(
            [
                "team_player_stats",
                "season",
                "week",
                "def_game_id",
            ]
        )
        .head(30)
    )

    print("\nExample duplicate defense rows:")
    print(duplicate_examples.to_string(index=False))

def_df = def_df.drop_duplicates(
    subset=[
        "team_player_stats",
        "season",
        "week",
    ],
    keep="first",
).copy()

rows_removed = rows_before_deduplication - len(def_df)

remaining_duplicates = def_df.duplicated(
    subset=[
        "team_player_stats",
        "season",
        "week",
    ],
    keep=False,
).sum()

print(
    f"\nRemoved duplicate defense team-week rows: "
    f"{rows_removed:,}"
)

print(
    f"Remaining duplicate defense team-week rows: "
    f"{remaining_duplicates:,}"
)

if remaining_duplicates != 0:
    raise ValueError(
        "Defense team-week duplicates were not fully removed."
    )


# ============================================================
# HISTORICAL FEATURES
# ============================================================

stat_columns = [
    TARGET,
    "points_allowed",
    "team_points",
    "def_sacks",
    "def_interceptions",
    "fumble_recovery_opp",
    "def_tds",
    "def_safeties",
    "def_qb_hits",
    "def_pass_defended",
    "def_tackles_for_loss",
    "blocked_kicks",
]

features = []

for column in stat_columns:
    last_game_feature = f"{column}_last_game"

    def_df[last_game_feature] = def_df.groupby(
        "team_player_stats",
        sort=False,
    )[column].shift(1)

    features.append(last_game_feature)

    for window in [
        3,
        5,
    ]:
        rolling_feature = f"{column}_last_{window}_avg"

        def_df[rolling_feature] = def_df.groupby(
            "team_player_stats",
            sort=False,
        )[column].transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    window=window,
                    min_periods=1,
                )
                .mean()
            )
        )

        features.append(rolling_feature)

season_stat_columns = [
    TARGET,
    "points_allowed",
    "def_sacks",
    "def_interceptions",
    "fumble_recovery_opp",
    "def_tds",
    "def_qb_hits",
]

for column in season_stat_columns:
    feature_name = f"{column}_season_avg"

    def_df[feature_name] = def_df.groupby(
        [
            "team_player_stats",
            "season",
        ],
        sort=False,
    )[column].transform(
        lambda values: (
            values.shift(1)
            .expanding(
                min_periods=1,
            )
            .mean()
        )
    )

    features.append(feature_name)

for column in [
    TARGET,
    "points_allowed",
    "def_sacks",
    "def_interceptions",
]:
    feature_name = f"{column}_ewma_3"

    def_df[feature_name] = def_df.groupby(
        "team_player_stats",
        sort=False,
    )[column].transform(
        lambda values: (
            values.shift(1)
            .ewm(
                span=3,
                adjust=False,
                min_periods=1,
            )
            .mean()
        )
    )

    features.append(feature_name)


# ============================================================
# PREVIOUS-SEASON FEATURES
# ============================================================

previous_season_columns = [
    TARGET,
    "points_allowed",
    "def_sacks",
    "def_interceptions",
    "fumble_recovery_opp",
    "def_tds",
]

previous_season_df = def_df.groupby(
    [
        "team_player_stats",
        "season",
    ],
    as_index=False,
).agg(
    **{
        f"{column}_prev_season": (
            column,
            "mean",
        )
        for column in previous_season_columns
    }
)

previous_season_df["season"] = previous_season_df["season"] + 1

def_df = def_df.merge(
    previous_season_df,
    on=[
        "team_player_stats",
        "season",
    ],
    how="left",
    validate="m:1",
)

previous_features = [f"{column}_prev_season" for column in previous_season_columns]

features.extend(previous_features)

def_df = def_df.sort_values(
    [
        "team_player_stats",
        "season",
        "week",
        "def_game_id",
    ],
    kind="mergesort",
).copy()


# ============================================================
# SAMPLE-SIZE FEATURES
# ============================================================

def_df["prior_team_games"] = def_df.groupby(
    "team_player_stats",
    sort=False,
).cumcount()

def_df["prior_team_games_this_season"] = def_df.groupby(
    [
        "team_player_stats",
        "season",
    ],
    sort=False,
).cumcount()

def_df["is_first_team_game"] = def_df["prior_team_games"].eq(0).astype(int)

def_df["is_first_team_game_of_season"] = (
    def_df["prior_team_games_this_season"].eq(0).astype(int)
)

features.extend(
    [
        "prior_team_games",
        "prior_team_games_this_season",
        "is_first_team_game",
        "is_first_team_game_of_season",
    ]
)


# ============================================================
# FILL HISTORICAL NULLS
# ============================================================

non_previous_features = [
    feature for feature in features if feature not in previous_features
]

def_df[non_previous_features] = def_df[non_previous_features].fillna(0)

for column in previous_season_columns:
    previous_feature = f"{column}_prev_season"

    season_feature = f"{column}_season_avg"

    if season_feature in def_df.columns:
        def_df[previous_feature] = (
            def_df[previous_feature].fillna(def_df[season_feature]).fillna(0)
        )

    else:
        def_df[previous_feature] = def_df[previous_feature].fillna(0)


# ============================================================
# CONTEXT FEATURES
# ============================================================

context_numeric_features = [
    "season",
    "week",
    "is_home",
    "team_rest",
    "div_game_player_stats",
    "temp_player_stats",
    "wind_player_stats",
    "spread_line_player_stats",
    "total_line_player_stats",
]

context_categorical_features = [
    "team_player_stats",
    "opponent_team",
    "roof_player_stats",
    "surface_player_stats",
]

missing_indicator_columns = [
    "temp_player_stats",
    "wind_player_stats",
    "spread_line_player_stats",
    "total_line_player_stats",
]

missing_features = []

for column in missing_indicator_columns:
    feature_name = f"{column}_missing"

    def_df[feature_name] = def_df[column].isna().astype(int)

    missing_features.append(feature_name)

for column in context_categorical_features:
    def_df[column] = (
        def_df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace(
            {
                "": "Unknown",
                "nan": "Unknown",
                "None": "Unknown",
            }
        )
    )

features = list(
    dict.fromkeys(
        features
        + context_numeric_features
        + context_categorical_features
        + missing_features
    )
)


# ============================================================
# MODEL DATA
# ============================================================

model_df = def_df.loc[def_df["season"].ge(2001)].copy()

safe_to_csv(
    model_df[
        [
            "def_game_id",
            "season",
            "week",
            "team_player_stats",
            "opponent_team",
        ]
        + features
        + [TARGET]
    ],
    MODEL_DATA_FILE,
)


# ============================================================
# BETTING VERSUS NO BETTING
# ============================================================

betting_features = {
    "spread_line_player_stats",
    "total_line_player_stats",
    "spread_line_player_stats_missing",
    "total_line_player_stats_missing",
}

feature_sets = {
    "def_without_betting": [
        feature for feature in features if feature not in betting_features
    ],
    "def_with_betting": features.copy(),
}


# ============================================================
# TIME SPLIT
# ============================================================

train_df = model_df.loc[model_df["season"].lt(VALIDATION_SEASON)].copy()

validation_df = model_df.loc[model_df["season"].eq(VALIDATION_SEASON)].copy()

test_df = model_df.loc[model_df["season"].eq(TEST_SEASON)].copy()

print("\nDefense dataset split:")
print(f"Training:   {len(train_df):,}")
print(f"Validation: {len(validation_df):,}")
print(f"Test:       {len(test_df):,}")

if train_df.empty:
    raise ValueError("Defense training data is empty.")

if validation_df.empty:
    raise ValueError("No defense rows exist for 2024.")

if test_df.empty:
    raise ValueError("No defense rows exist for 2025.")


# ============================================================
# TRAIN VALIDATION MODELS
# ============================================================

validation_results = []

for model_name, model_features in feature_sets.items():
    print("\n" + "-" * 70)
    print(f"Training {model_name}")
    print(f"Feature count: {len(model_features)}")

    pipeline = build_random_forest_pipeline(
        dataframe=train_df,
        features=model_features,
        categorical_features=(context_categorical_features),
    )

    pipeline.fit(
        train_df[model_features],
        train_df[TARGET],
    )

    validation_predictions = pipeline.predict(validation_df[model_features])

    metrics = calculate_metrics(
        validation_df[TARGET],
        validation_predictions,
    )

    validation_results.append(
        {
            "model": model_name,
            "feature_count": len(model_features),
            "validation_mae": metrics["mae"],
            "validation_rmse": metrics["rmse"],
            "validation_r2": metrics["r2"],
        }
    )

    print(f"MAE:  {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"R²:   {metrics['r2']:.3f}")

results_df = (
    pd.DataFrame(validation_results)
    .sort_values("validation_mae")
    .reset_index(drop=True)
)

print("\nValidation results:")
print(results_df.to_string(index=False))

safe_to_csv(
    results_df,
    RESULTS_FILE,
)


# ============================================================
# CHOOSE BETTING ONLY IF IT MEANINGFULLY IMPROVES MAE
# ============================================================

without_betting_mae = float(
    results_df.loc[
        results_df["model"].eq("def_without_betting"),
        "validation_mae",
    ].iloc[0]
)

with_betting_mae = float(
    results_df.loc[
        results_df["model"].eq("def_with_betting"),
        "validation_mae",
    ].iloc[0]
)

betting_improvement = without_betting_mae - with_betting_mae

if betting_improvement >= MIN_BETTING_MAE_IMPROVEMENT:
    best_model_name = "def_with_betting"

else:
    best_model_name = "def_without_betting"

best_features = feature_sets[best_model_name]

print(f"\nSelected defense model: {best_model_name}")


# ============================================================
# FINAL MODEL AND 2025 TEST
# ============================================================

final_training_df = model_df.loc[model_df["season"].lt(TEST_SEASON)].copy()

final_model = build_random_forest_pipeline(
    dataframe=final_training_df,
    features=best_features,
    categorical_features=(context_categorical_features),
)

final_model.fit(
    final_training_df[best_features],
    final_training_df[TARGET],
)

test_predictions = final_model.predict(test_df[best_features])

test_metrics = calculate_metrics(
    test_df[TARGET],
    test_predictions,
)

print("\nFinal 2025 defense results:")
print(f"MAE:  {test_metrics['mae']:.3f}")
print(f"RMSE: {test_metrics['rmse']:.3f}")
print(f"R²:   {test_metrics['r2']:.3f}")


# ============================================================
# SAVE OUTPUTS
# ============================================================

predictions_df = pd.DataFrame(
    {
        "player_name": test_df["team_player_stats"].values,
        "season": test_df["season"].values,
        "week": test_df["week"].values,
        "predicted_fantasy_points": test_predictions,
        "position": "DEF",
    }
)

predictions_df["predicted_fantasy_points"] = (
    predictions_df["predicted_fantasy_points"].round(2)
)

predictions_df = predictions_df[
    [
        "player_name",
        "season",
        "week",
        "predicted_fantasy_points",
        "position",
    ]
]

safe_to_csv(
    predictions_df,
    PREDICTIONS_FILE,
)

joblib.dump(
    {
        "model": final_model,
        "features": best_features,
        "target": TARGET,
        "feature_set": best_model_name,
        "test_metrics": test_metrics,
    },
    MODEL_FILE,
)

print(f"Saved model: {MODEL_FILE}")
print("DEFENSE MODEL FINISHED")
