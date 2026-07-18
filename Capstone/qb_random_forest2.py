from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# SETTINGS
# This file should be located inside the Capstone folder.
#
# Expected input:
# Capstone/data/main_df_with_sleeper_ids.csv
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"

INPUT_FILE = DATA_DIR / "main_df_with_sleeper_ids.csv"
MODEL_DATA_FILE = DATA_DIR / "qb_model_dataset.csv"
RESULTS_FILE = DATA_DIR / "qb_random_forest_results.csv"
PREDICTIONS_FILE = DATA_DIR / "qb_random_forest_2025_predictions.csv"
MODEL_FILE = DATA_DIR / "qb_random_forest_model.joblib"

TARGET = "fantasy_points"

VALIDATION_SEASON = 2024
TEST_SEASON = 2025

RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def safe_to_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save a CSV.

    If Windows blocks the file because it is open somewhere,
    save a timestamped version instead.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        dataframe.to_csv(
            path,
            index=False,
        )

        print(f"Saved: {path}")

    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        backup_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

        dataframe.to_csv(
            backup_path,
            index=False,
        )

        print(f"{path} was locked. Saved instead: {backup_path}")


def require_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Stop the script with a readable message if required
    columns are missing.
    """

    missing_columns = [column for column in columns if column not in dataframe.columns]

    if missing_columns:
        raise KeyError(
            "The dataset is missing these required columns:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    """
    Calculate regression evaluation metrics.
    """

    return {
        "mae": mean_absolute_error(
            actual,
            predicted,
        ),
        "rmse": mean_squared_error(
            actual,
            predicted,
        )
        ** 0.5,
        "r2": r2_score(
            actual,
            predicted,
        ),
    }


def build_random_forest_pipeline(
    dataframe: pd.DataFrame,
    features: list[str],
) -> Pipeline:
    """
    Create the preprocessing and Random Forest pipeline.
    """

    categorical_features = [
        feature
        for feature in features
        if (
            pd.api.types.is_object_dtype(dataframe[feature])
            or pd.api.types.is_string_dtype(dataframe[feature])
            or isinstance(
                dataframe[feature].dtype,
                pd.CategoricalDtype,
            )
        )
    ]

    numeric_features = [
        feature for feature in features if feature not in categorical_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    random_forest = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                random_forest,
            ),
        ]
    )


# ============================================================
# LOAD MAIN DATASET
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find the input file:\n"
        f"{INPUT_FILE}\n\n"
        "Place this Python file inside the Capstone folder and "
        "make sure main_df_with_sleeper_ids.csv is inside "
        "Capstone/data."
    )

print(f"Loading: {INPUT_FILE}")

main_df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

print(f"Loaded rows: {len(main_df):,}")


# ============================================================
# CHECK REQUIRED RAW COLUMNS
# ============================================================

REQUIRED_RAW_COLUMNS = [
    "player_id",
    "position_player_stats",
    "season",
    "week",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "attempts",
    TARGET,
]

require_columns(
    main_df,
    REQUIRED_RAW_COLUMNS,
)


# ============================================================
# FILTER TO QUARTERBACKS
# ============================================================

qb_df = main_df.loc[main_df["position_player_stats"].eq("QB")].copy()

qb_df = qb_df.dropna(
    subset=[
        "player_id",
        "season",
        "week",
        TARGET,
    ]
).copy()

qb_df["player_id"] = qb_df["player_id"].astype(str).str.strip()

print(f"QB rows: {len(qb_df):,}")


# ============================================================
# CREATE OPTIONAL COLUMNS IF THEY ARE NOT AVAILABLE
# ============================================================

OPTIONAL_NUMERIC_COLUMNS = [
    "draft_year",
    "temp_player_stats",
    "wind_player_stats",
    "spread_line_player_stats",
    "total_line_player_stats",
]

OPTIONAL_CATEGORICAL_COLUMNS = [
    "team_player_stats",
    "opponent_team",
    "away_team_player_stats",
    "home_team_player_stats",
    "roof_player_stats",
    "surface_player_stats",
]

for column in OPTIONAL_NUMERIC_COLUMNS + OPTIONAL_CATEGORICAL_COLUMNS:
    if column not in qb_df.columns:
        qb_df[column] = np.nan

        print(f"Created unavailable optional column: {column}")


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS = [
    "season",
    "week",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "attempts",
    TARGET,
    *OPTIONAL_NUMERIC_COLUMNS,
]

for column in NUMERIC_COLUMNS:
    qb_df[column] = pd.to_numeric(
        qb_df[column],
        errors="coerce",
    )

qb_df = qb_df.dropna(
    subset=[
        "season",
        "week",
        TARGET,
    ]
).copy()

qb_df["season"] = qb_df["season"].astype(int)

qb_df["week"] = qb_df["week"].astype(int)


# ============================================================
# FILL OPPONENT_TEAM
#
# If the player's team is the away team, opponent is home.
# If the player's team is the home team, opponent is away.
# ============================================================

derived_opponent = np.select(
    [
        qb_df["team_player_stats"].eq(qb_df["away_team_player_stats"]),
        qb_df["team_player_stats"].eq(qb_df["home_team_player_stats"]),
    ],
    [
        qb_df["home_team_player_stats"],
        qb_df["away_team_player_stats"],
    ],
    default=None,
)

derived_opponent = pd.Series(
    derived_opponent,
    index=qb_df.index,
    dtype="object",
)

opponent_missing = qb_df["opponent_team"].isna() | qb_df["opponent_team"].astype(
    str
).str.strip().isin(
    [
        "",
        "nan",
        "None",
    ]
)

qb_df.loc[
    opponent_missing,
    "opponent_team",
] = derived_opponent.loc[opponent_missing]


# ============================================================
# SORT DATA CHRONOLOGICALLY
# ============================================================

SORT_COLUMNS = [
    "player_id",
    "season",
    "week",
]

if "gameday_player_stats" in qb_df.columns:
    qb_df["gameday_player_stats"] = pd.to_datetime(
        qb_df["gameday_player_stats"],
        errors="coerce",
    )

    SORT_COLUMNS.append("gameday_player_stats")

qb_df = qb_df.sort_values(
    SORT_COLUMNS,
    kind="mergesort",
).copy()


# ============================================================
# FEATURE ENGINEERING FUNCTIONS
#
# shift(1) ensures the current game is never included in its
# own historical features.
# ============================================================


def add_last_game_feature(
    column: str,
) -> None:
    qb_df[f"{column}_last_game"] = qb_df.groupby(
        "player_id",
        sort=False,
    )[column].shift(1)


def add_rolling_average(
    column: str,
    window: int,
) -> None:
    qb_df[f"{column}_last_{window}_avg"] = qb_df.groupby(
        "player_id",
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


def add_season_average(
    column: str,
) -> None:
    qb_df[f"{column}_season_avg"] = qb_df.groupby(
        [
            "player_id",
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


def add_career_average(
    column: str,
) -> None:
    qb_df[f"{column}_career_avg"] = qb_df.groupby(
        "player_id",
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


def add_ewma(
    column: str,
    span: int = 3,
) -> None:
    qb_df[f"{column}_ewma_{span}"] = qb_df.groupby(
        "player_id",
        sort=False,
    )[column].transform(
        lambda values: (
            values.shift(1)
            .ewm(
                span=span,
                adjust=False,
                min_periods=1,
            )
            .mean()
        )
    )


# ============================================================
# LAST-GAME FEATURES
# ============================================================

add_last_game_feature("passing_yards")

add_last_game_feature(TARGET)


# ============================================================
# ROLLING-AVERAGE FEATURES
# ============================================================

ROLLING_FEATURE_CONFIG = [
    (
        "passing_yards",
        3,
    ),
    (
        "passing_tds",
        3,
    ),
    (
        TARGET,
        5,
    ),
    (
        "rushing_yards",
        3,
    ),
    (
        "attempts",
        3,
    ),
]

for column, window in ROLLING_FEATURE_CONFIG:
    add_rolling_average(
        column,
        window,
    )


# ============================================================
# SEASON AND CAREER AVERAGES
# ============================================================

BASE_STATISTICS = [
    TARGET,
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "attempts",
]

for column in BASE_STATISTICS:
    add_season_average(column)

    add_career_average(column)


# ============================================================
# EWMA FEATURES
# ============================================================

for column in [
    TARGET,
    "passing_yards",
    "passing_tds",
    "rushing_yards",
]:
    add_ewma(
        column,
        span=3,
    )


# ============================================================
# PREVIOUS-SEASON AVERAGES
# ============================================================

previous_season_df = qb_df.groupby(
    [
        "player_id",
        "season",
    ],
    as_index=False,
).agg(
    fantasy_points_prev_season=(
        TARGET,
        "mean",
    ),
    passing_yards_prev_season=(
        "passing_yards",
        "mean",
    ),
    passing_tds_prev_season=(
        "passing_tds",
        "mean",
    ),
    rushing_yards_prev_season=(
        "rushing_yards",
        "mean",
    ),
    attempts_prev_season=(
        "attempts",
        "mean",
    ),
)

previous_season_df["season"] = previous_season_df["season"] + 1

qb_df = qb_df.merge(
    previous_season_df,
    on=[
        "player_id",
        "season",
    ],
    how="left",
    validate="m:1",
)

qb_df = qb_df.sort_values(
    SORT_COLUMNS,
    kind="mergesort",
).copy()


# ============================================================
# ROOKIE AND HISTORY FLAGS
# ============================================================

qb_df["career_game_number"] = (
    qb_df.groupby(
        "player_id",
        sort=False,
    )
    .cumcount()
    .add(1)
)

qb_df["season_game_number"] = (
    qb_df.groupby(
        [
            "player_id",
            "season",
        ],
        sort=False,
    )
    .cumcount()
    .add(1)
)

qb_df["is_first_observed_game"] = qb_df["career_game_number"].eq(1).astype(int)

qb_df["is_first_game_of_season"] = qb_df["season_game_number"].eq(1).astype(int)

qb_df["first_observed_season"] = qb_df.groupby(
    "player_id",
    sort=False,
)["season"].transform("min")

data_start_season = int(qb_df["season"].min())

qb_df["is_known_rookie_season"] = (
    qb_df["draft_year"].eq(qb_df["season"]).fillna(False).astype(int)
)

qb_df["is_possible_rookie_season"] = (
    qb_df["first_observed_season"].eq(qb_df["season"])
    & qb_df["season"].gt(data_start_season)
).astype(int)

qb_df["is_rookie_season"] = (
    qb_df["is_known_rookie_season"].eq(1)
    | (qb_df["draft_year"].isna() & qb_df["is_possible_rookie_season"].eq(1))
).astype(int)

qb_df["is_rookie_first_game"] = (
    qb_df["is_first_observed_game"].eq(1) & qb_df["is_rookie_season"].eq(1)
).astype(int)

qb_df["has_prior_qb_game"] = qb_df["is_first_observed_game"].eq(0).astype(int)


# ============================================================
# FILL ENGINEERED HISTORY NULLS
# ============================================================

HISTORY_FEATURES = [
    "passing_yards_last_game",
    "fantasy_points_last_game",
    "passing_yards_last_3_avg",
    "passing_tds_last_3_avg",
    "fantasy_points_last_5_avg",
    "rushing_yards_last_3_avg",
    "attempts_last_3_avg",
    "fantasy_points_career_avg",
    "passing_yards_career_avg",
    "passing_tds_career_avg",
    "rushing_yards_career_avg",
    "attempts_career_avg",
    "fantasy_points_ewma_3",
    "passing_yards_ewma_3",
    "passing_tds_ewma_3",
    "rushing_yards_ewma_3",
]

SEASON_AVERAGE_FEATURES = [
    "fantasy_points_season_avg",
    "passing_yards_season_avg",
    "passing_tds_season_avg",
    "rushing_yards_season_avg",
    "attempts_season_avg",
]

qb_df[HISTORY_FEATURES] = qb_df[HISTORY_FEATURES].fillna(0)

qb_df[SEASON_AVERAGE_FEATURES] = qb_df[SEASON_AVERAGE_FEATURES].fillna(0)


# ============================================================
# FILL PREVIOUS-SEASON NULLS
# ============================================================

PREVIOUS_TO_CAREER_MAP = {
    "fantasy_points_prev_season": "fantasy_points_career_avg",
    "passing_yards_prev_season": "passing_yards_career_avg",
    "passing_tds_prev_season": "passing_tds_career_avg",
    "rushing_yards_prev_season": "rushing_yards_career_avg",
    "attempts_prev_season": "attempts_career_avg",
}

for (
    previous_column,
    career_column,
) in PREVIOUS_TO_CAREER_MAP.items():
    qb_df[previous_column] = (
        qb_df[previous_column].fillna(qb_df[career_column]).fillna(0)
    )


# ============================================================
# CREATE MISSING-VALUE FLAGS
#
# Do not fill numeric context columns here.
# The sklearn pipeline calculates medians from training data
# only, preventing validation/test leakage.
# ============================================================

CONTEXT_NUMERIC_COLUMNS = [
    "temp_player_stats",
    "wind_player_stats",
    "spread_line_player_stats",
    "total_line_player_stats",
]

for column in CONTEXT_NUMERIC_COLUMNS:
    qb_df[f"{column}_missing"] = qb_df[column].isna().astype(int)


# ============================================================
# CLEAN CATEGORICAL COLUMNS
# ============================================================

CATEGORICAL_CONTEXT_COLUMNS = [
    "team_player_stats",
    "opponent_team",
    "roof_player_stats",
    "surface_player_stats",
]

for column in CATEGORICAL_CONTEXT_COLUMNS:
    qb_df[column] = (
        qb_df[column]
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


# ============================================================
# DEFINE MODEL FEATURES
# ============================================================

BASELINE_FEATURES = [
    # Recent performance
    "passing_yards_last_game",
    "fantasy_points_last_game",
    "passing_yards_last_3_avg",
    "passing_tds_last_3_avg",
    "fantasy_points_last_5_avg",
    "rushing_yards_last_3_avg",
    "attempts_last_3_avg",
    # Current-season averages
    "fantasy_points_season_avg",
    "passing_yards_season_avg",
    "passing_tds_season_avg",
    "rushing_yards_season_avg",
    "attempts_season_avg",
    # Career averages
    "fantasy_points_career_avg",
    "passing_yards_career_avg",
    "passing_tds_career_avg",
    "rushing_yards_career_avg",
    "attempts_career_avg",
    # Previous-season averages
    "fantasy_points_prev_season",
    "passing_yards_prev_season",
    "passing_tds_prev_season",
    "rushing_yards_prev_season",
    "attempts_prev_season",
    # Exponentially weighted averages
    "fantasy_points_ewma_3",
    "passing_yards_ewma_3",
    "passing_tds_ewma_3",
    "rushing_yards_ewma_3",
    # Rookie/history flags
    "is_first_observed_game",
    "is_first_game_of_season",
    "is_rookie_season",
    "is_rookie_first_game",
    "has_prior_qb_game",
    # Game context
    "season",
    "week",
    "team_player_stats",
    "opponent_team",
    "roof_player_stats",
    "surface_player_stats",
    "temp_player_stats",
    "wind_player_stats",
    "spread_line_player_stats",
    "total_line_player_stats",
    # Missing-value flags
    "temp_player_stats_missing",
    "wind_player_stats_missing",
    "spread_line_player_stats_missing",
    "total_line_player_stats_missing",
]

require_columns(
    qb_df,
    BASELINE_FEATURES + [TARGET],
)


# ============================================================
# CREATE MODEL DATASET
#
# The 2000 season is only used to create history for 2001.
# ============================================================

model_df = qb_df.loc[qb_df["season"].ge(2001)].copy()

debug_columns = [
    column
    for column in [
        "player_id",
        "player_display_name",
        "player_name",
    ]
    if column in model_df.columns
]

qb_model_output = model_df[debug_columns + BASELINE_FEATURES + [TARGET]].copy()

safe_to_csv(
    qb_model_output,
    MODEL_DATA_FILE,
)


# ============================================================
# CREATE BETTING AND NO-BETTING FEATURE SETS
# ============================================================

BETTING_FEATURES = {
    "spread_line_player_stats",
    "total_line_player_stats",
    "spread_line_player_stats_missing",
    "total_line_player_stats_missing",
}

FEATURE_SETS = {
    "random_forest_without_betting": [
        feature for feature in BASELINE_FEATURES if feature not in BETTING_FEATURES
    ],
    "random_forest_with_betting": BASELINE_FEATURES.copy(),
}


# ============================================================
# DISPLAY BETTING MISSINGNESS
# ============================================================

print("\nBetting data missingness:")

for column in [
    "spread_line_player_stats",
    "total_line_player_stats",
]:
    missing_percentage = model_df[column].isna().mean() * 100

    print(f"{column}: {missing_percentage:.1f}%")


# ============================================================
# TIME-BASED SPLIT
#
# Training: seasons before 2024
# Validation: 2024
# Final test: 2025
# ============================================================

train_df = model_df.loc[model_df["season"].lt(VALIDATION_SEASON)].copy()

validation_df = model_df.loc[model_df["season"].eq(VALIDATION_SEASON)].copy()

test_df = model_df.loc[model_df["season"].eq(TEST_SEASON)].copy()

if train_df.empty:
    raise ValueError(f"No training rows exist before {VALIDATION_SEASON}.")

if validation_df.empty:
    raise ValueError(f"No validation rows exist for {VALIDATION_SEASON}.")

if test_df.empty:
    raise ValueError(f"No test rows exist for {TEST_SEASON}.")

print("\nDataset split:")

print(f"Training rows before {VALIDATION_SEASON}: {len(train_df):,}")

print(f"Validation rows for {VALIDATION_SEASON}: {len(validation_df):,}")

print(f"Test rows for {TEST_SEASON}: {len(test_df):,}")


# ============================================================
# COMPARE BETTING VS. NO-BETTING RANDOM FORESTS
# ============================================================

validation_results = []

for (
    model_name,
    model_features,
) in FEATURE_SETS.items():
    print(f"\nTraining {model_name}...")

    pipeline = build_random_forest_pipeline(
        train_df,
        model_features,
    )

    pipeline.fit(
        train_df[model_features],
        train_df[TARGET],
    )

    validation_predictions = pipeline.predict(validation_df[model_features])

    model_metrics = calculate_metrics(
        validation_df[TARGET],
        validation_predictions,
    )

    validation_results.append(
        {
            "model": model_name,
            "feature_count": len(model_features),
            "validation_mae": model_metrics["mae"],
            "validation_rmse": model_metrics["rmse"],
            "validation_r2": model_metrics["r2"],
        }
    )

    print(f"MAE:  {model_metrics['mae']:.3f}")

    print(f"RMSE: {model_metrics['rmse']:.3f}")

    print(f"R²:   {model_metrics['r2']:.3f}")


validation_results_df = (
    pd.DataFrame(validation_results)
    .sort_values(
        "validation_mae",
        ascending=True,
    )
    .reset_index(drop=True)
)

print("\nValidation comparison:")

print(validation_results_df.to_string(index=False))

safe_to_csv(
    validation_results_df,
    RESULTS_FILE,
)


# ============================================================
# SELECT THE BETTER FEATURE SET
# ============================================================

best_model_name = str(
    validation_results_df.loc[
        0,
        "model",
    ]
)

best_features = FEATURE_SETS[best_model_name]

print(f"\nSelected feature set: {best_model_name}")

print(f"Selected feature count: {len(best_features)}")


# ============================================================
# RETRAIN THE WINNER USING DATA THROUGH 2024
# ============================================================

final_training_df = model_df.loc[model_df["season"].lt(TEST_SEASON)].copy()

final_model = build_random_forest_pipeline(
    final_training_df,
    best_features,
)

final_model.fit(
    final_training_df[best_features],
    final_training_df[TARGET],
)


# ============================================================
# FINAL 2025 TEST
# ============================================================

test_predictions = final_model.predict(test_df[best_features])

test_metrics = calculate_metrics(
    test_df[TARGET],
    test_predictions,
)

print("\nFinal 2025 test results:")

print(f"Model: {best_model_name}")

print(f"MAE:  {test_metrics['mae']:.3f}")

print(f"RMSE: {test_metrics['rmse']:.3f}")

print(f"R²:   {test_metrics['r2']:.3f}")


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_columns = [
    column
    for column in [
        "player_id",
        "player_display_name",
        "player_name",
        "season",
        "week",
        "team_player_stats",
        "opponent_team",
        TARGET,
    ]
    if column in test_df.columns
]

predictions_df = test_df[prediction_columns].copy()

predictions_df["predicted_fantasy_points"] = test_predictions

predictions_df["prediction_error"] = (
    predictions_df[TARGET] - predictions_df["predicted_fantasy_points"]
)

predictions_df["absolute_error"] = predictions_df["prediction_error"].abs()

predictions_df = predictions_df.sort_values(
    [
        "season",
        "week",
        "absolute_error",
    ],
    ascending=[
        True,
        True,
        False,
    ],
)

safe_to_csv(
    predictions_df,
    PREDICTIONS_FILE,
)


# ============================================================
# SAVE MODEL
# ============================================================

model_bundle = {
    "model": final_model,
    "features": best_features,
    "selected_feature_set": best_model_name,
    "target": TARGET,
    "validation_season": VALIDATION_SEASON,
    "test_season": TEST_SEASON,
    "test_metrics": test_metrics,
}

joblib.dump(
    model_bundle,
    MODEL_FILE,
)

print(f"Saved model: {MODEL_FILE}")

print("\nFINISHED SUCCESSFULLY.")

print(f"Model dataset: {MODEL_DATA_FILE}")

print(f"Validation results: {RESULTS_FILE}")

print(f"Test predictions: {PREDICTIONS_FILE}")

print(f"Saved model: {MODEL_FILE}")
