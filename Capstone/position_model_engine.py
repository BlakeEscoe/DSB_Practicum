from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_INPUT_FILE = DATA_DIR / "main_df_with_sleeper_ids.csv"


def safe_to_csv(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        dataframe.to_csv(
            output_path,
            index=False,
        )
        print(f"Saved: {output_path}")
        return output_path

    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        backup_path = output_path.with_name(
            f"{output_path.stem}_{timestamp}{output_path.suffix}"
        )

        dataframe.to_csv(
            backup_path,
            index=False,
        )

        print(f"{output_path.name} was locked. Saved instead as {backup_path.name}")

        return backup_path


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        formatted_columns = "\n".join(f"- {column}" for column in missing_columns)

        raise KeyError(
            f"The dataframe is missing required columns:\n{formatted_columns}"
        )


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
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
    categorical_features: list[str],
    random_state: int = 42,
) -> Pipeline:
    categorical_features = [
        feature for feature in categorical_features if feature in features
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
                "one_hot",
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

    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def fill_opponent_team(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    needed_columns = {
        "team_player_stats",
        "opponent_team",
        "away_team_player_stats",
        "home_team_player_stats",
    }

    if not needed_columns.issubset(dataframe.columns):
        return dataframe

    derived_opponent = np.select(
        [
            dataframe["team_player_stats"].eq(dataframe["away_team_player_stats"]),
            dataframe["team_player_stats"].eq(dataframe["home_team_player_stats"]),
        ],
        [
            dataframe["home_team_player_stats"],
            dataframe["away_team_player_stats"],
        ],
        default=None,
    )

    derived_opponent = pd.Series(
        derived_opponent,
        index=dataframe.index,
        dtype="object",
    )

    opponent_missing = dataframe["opponent_team"].isna() | dataframe[
        "opponent_team"
    ].astype(str).str.strip().isin(
        [
            "",
            "nan",
            "None",
        ]
    )

    dataframe.loc[
        opponent_missing,
        "opponent_team",
    ] = derived_opponent.loc[opponent_missing]

    return dataframe


def run_player_position_model(
    config: dict[str, Any],
) -> None:
    model_name = config["model_name"]
    position_value = config["position_value"]
    target = config["target"]

    validation_season = config.get(
        "validation_season",
        2024,
    )

    test_season = config.get(
        "test_season",
        2025,
    )

    minimum_model_season = config.get(
        "minimum_model_season",
        2001,
    )

    input_file = Path(
        config.get(
            "input_file",
            DEFAULT_INPUT_FILE,
        )
    )

    if not input_file.exists():
        raise FileNotFoundError(f"Could not find:\n{input_file}")

    print("=" * 70)
    print(f"LOADING {model_name.upper()} DATA")
    print("=" * 70)

    dataframe = pd.read_csv(
        input_file,
        low_memory=False,
    )

    require_columns(
        dataframe,
        [
            "player_id",
            "position_player_stats",
            "season",
            "week",
            target,
        ],
    )

    dataframe = dataframe.loc[
        dataframe["position_player_stats"].eq(position_value)
    ].copy()

    print(f"{position_value} rows before cleaning: {len(dataframe):,}")

    numeric_raw_columns = list(
        dict.fromkeys(
            config.get(
                "numeric_raw_columns",
                [],
            )
            + [
                "season",
                "week",
                "draft_year",
                target,
            ]
        )
    )

    categorical_raw_columns = list(
        dict.fromkeys(
            config.get(
                "categorical_raw_columns",
                [],
            )
            + [
                "team_player_stats",
                "opponent_team",
                "away_team_player_stats",
                "home_team_player_stats",
            ]
        )
    )

    for column in numeric_raw_columns:
        if column not in dataframe.columns:
            dataframe[column] = np.nan
            print(f"Created missing numeric column: {column}")

    for column in categorical_raw_columns:
        if column not in dataframe.columns:
            dataframe[column] = "Unknown"
            print(f"Created missing categorical column: {column}")

    for column in numeric_raw_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=[
            "player_id",
            "season",
            "week",
            target,
        ]
    ).copy()

    dataframe["player_id"] = dataframe["player_id"].astype(str).str.strip()

    dataframe["season"] = dataframe["season"].astype(int)

    dataframe["week"] = dataframe["week"].astype(int)

    dataframe = fill_opponent_team(dataframe)

    sort_columns = [
        "player_id",
        "season",
        "week",
    ]

    if "gameday_player_stats" in dataframe.columns:
        dataframe["gameday_player_stats"] = pd.to_datetime(
            dataframe["gameday_player_stats"],
            errors="coerce",
        )

        sort_columns.append("gameday_player_stats")

    dataframe = dataframe.sort_values(
        sort_columns,
        kind="mergesort",
    ).copy()

    derived_builder: (
        Callable[
            [pd.DataFrame],
            pd.DataFrame,
        ]
        | None
    ) = config.get("derived_builder")

    if derived_builder is not None:
        dataframe = derived_builder(dataframe)

    generated_features: list[str] = []

    last_game_columns = config.get(
        "last_game_columns",
        [],
    )

    for column in last_game_columns:
        require_columns(
            dataframe,
            [column],
        )

        feature_name = f"{column}_last_game"

        dataframe[feature_name] = dataframe.groupby(
            "player_id",
            sort=False,
        )[column].shift(1)

        generated_features.append(feature_name)

    rolling_windows = config.get(
        "rolling_windows",
        {},
    )

    for column, windows in rolling_windows.items():
        require_columns(
            dataframe,
            [column],
        )

        for window in windows:
            feature_name = f"{column}_last_{window}_avg"

            dataframe[feature_name] = dataframe.groupby(
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

            generated_features.append(feature_name)

    season_average_columns = config.get(
        "season_average_columns",
        [],
    )

    for column in season_average_columns:
        require_columns(
            dataframe,
            [column],
        )

        feature_name = f"{column}_season_avg"

        dataframe[feature_name] = dataframe.groupby(
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

        generated_features.append(feature_name)

    career_average_columns = config.get(
        "career_average_columns",
        [],
    )

    for column in career_average_columns:
        require_columns(
            dataframe,
            [column],
        )

        feature_name = f"{column}_career_avg"

        dataframe[feature_name] = dataframe.groupby(
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

        generated_features.append(feature_name)

    previous_season_columns = config.get(
        "previous_season_columns",
        [],
    )

    previous_feature_names: list[str] = []

    if previous_season_columns:
        aggregation_dictionary = {
            f"{column}_prev_season": (
                column,
                "mean",
            )
            for column in previous_season_columns
        }

        previous_season_df = dataframe.groupby(
            [
                "player_id",
                "season",
            ],
            as_index=False,
        ).agg(**aggregation_dictionary)

        previous_season_df["season"] = previous_season_df["season"] + 1

        dataframe = dataframe.merge(
            previous_season_df,
            on=[
                "player_id",
                "season",
            ],
            how="left",
            validate="m:1",
        )

        previous_feature_names = list(aggregation_dictionary.keys())

        generated_features.extend(previous_feature_names)

        dataframe = dataframe.sort_values(
            sort_columns,
            kind="mergesort",
        ).copy()

    ewma_columns = config.get(
        "ewma_columns",
        [],
    )

    ewma_span = config.get(
        "ewma_span",
        3,
    )

    for column in ewma_columns:
        require_columns(
            dataframe,
            [column],
        )

        feature_name = f"{column}_ewma_{ewma_span}"

        dataframe[feature_name] = dataframe.groupby(
            "player_id",
            sort=False,
        )[column].transform(
            lambda values: (
                values.shift(1)
                .ewm(
                    span=ewma_span,
                    adjust=False,
                    min_periods=1,
                )
                .mean()
            )
        )

        generated_features.append(feature_name)

    dataframe["prior_career_games"] = dataframe.groupby(
        "player_id",
        sort=False,
    ).cumcount()

    dataframe["prior_games_this_season"] = dataframe.groupby(
        [
            "player_id",
            "season",
        ],
        sort=False,
    ).cumcount()

    dataframe["is_first_observed_game"] = (
        dataframe["prior_career_games"].eq(0).astype(int)
    )

    dataframe["is_first_game_of_season"] = (
        dataframe["prior_games_this_season"].eq(0).astype(int)
    )

    dataframe["first_observed_season"] = dataframe.groupby(
        "player_id",
        sort=False,
    )["season"].transform("min")

    earliest_season = int(dataframe["season"].min())

    known_rookie = dataframe["draft_year"].eq(dataframe["season"]).fillna(False)

    possible_rookie = dataframe["first_observed_season"].eq(
        dataframe["season"]
    ) & dataframe["season"].gt(earliest_season)

    dataframe["is_rookie_season"] = (
        known_rookie | (dataframe["draft_year"].isna() & possible_rookie)
    ).astype(int)

    dataframe["is_rookie_first_game"] = (
        dataframe["is_first_observed_game"].eq(1) & dataframe["is_rookie_season"].eq(1)
    ).astype(int)

    dataframe["has_prior_position_game"] = (
        dataframe["prior_career_games"].gt(0).astype(int)
    )

    history_flags = [
        "prior_career_games",
        "prior_games_this_season",
        "is_first_observed_game",
        "is_first_game_of_season",
        "is_rookie_season",
        "is_rookie_first_game",
        "has_prior_position_game",
    ]

    generated_features.extend(history_flags)

    non_previous_features = [
        feature
        for feature in generated_features
        if feature not in previous_feature_names
    ]

    dataframe[non_previous_features] = dataframe[non_previous_features].fillna(0)

    for column in previous_season_columns:
        previous_feature = f"{column}_prev_season"

        career_feature = f"{column}_career_avg"

        if career_feature in dataframe.columns:
            dataframe[previous_feature] = (
                dataframe[previous_feature].fillna(dataframe[career_feature]).fillna(0)
            )

        else:
            dataframe[previous_feature] = dataframe[previous_feature].fillna(0)

    context_numeric_columns = config.get(
        "context_numeric_columns",
        [],
    )

    context_categorical_columns = config.get(
        "context_categorical_columns",
        [],
    )

    missing_indicator_columns = config.get(
        "missing_indicator_columns",
        [],
    )

    missing_features: list[str] = []

    for column in missing_indicator_columns:
        if column not in dataframe.columns:
            dataframe[column] = np.nan

        feature_name = f"{column}_missing"

        dataframe[feature_name] = dataframe[column].isna().astype(int)

        missing_features.append(feature_name)

    for column in context_categorical_columns:
        if column not in dataframe.columns:
            dataframe[column] = "Unknown"

        dataframe[column] = (
            dataframe[column]
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

    all_features = list(
        dict.fromkeys(
            generated_features
            + context_numeric_columns
            + context_categorical_columns
            + missing_features
        )
    )

    require_columns(
        dataframe,
        all_features + [target],
    )

    model_df = dataframe.loc[dataframe["season"].ge(minimum_model_season)].copy()

    betting_columns = config.get(
        "betting_columns",
        [],
    )

    betting_feature_names = set(
        betting_columns + [f"{column}_missing" for column in betting_columns]
    )

    feature_sets = {
        f"{model_name}_without_betting": [
            feature for feature in all_features if feature not in betting_feature_names
        ],
        f"{model_name}_with_betting": all_features.copy(),
    }

    output_prefix = config.get(
        "output_prefix",
        model_name.lower(),
    )

    model_dataset_file = DATA_DIR / f"{output_prefix}_model_dataset.csv"

    results_file = DATA_DIR / f"{output_prefix}_random_forest_results.csv"

    predictions_file = DATA_DIR / f"{output_prefix}_2025_predictions.csv"

    model_file = DATA_DIR / f"{output_prefix}_random_forest_model.joblib"

    identifier_columns = [
        column
        for column in [
            "player_id",
            "player_display_name",
            "player_name",
            "position_player_stats",
            "season",
            "week",
            "team_player_stats",
            "opponent_team",
        ]
        if column in model_df.columns
    ]

    model_output = model_df[
        list(dict.fromkeys(identifier_columns + all_features + [target]))
    ].copy()

    safe_to_csv(
        model_output,
        model_dataset_file,
    )

    train_df = model_df.loc[model_df["season"].lt(validation_season)].copy()

    validation_df = model_df.loc[model_df["season"].eq(validation_season)].copy()

    test_df = model_df.loc[model_df["season"].eq(test_season)].copy()

    if train_df.empty:
        raise ValueError("The training dataframe is empty.")

    if validation_df.empty:
        raise ValueError(f"No rows found for validation season {validation_season}.")

    if test_df.empty:
        raise ValueError(f"No rows found for test season {test_season}.")

    print("\nDataset split")
    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows:       {len(test_df):,}")

    print("\nMissing betting data")

    for column in betting_columns:
        if column in model_df.columns:
            missing_percent = model_df[column].isna().mean() * 100

            print(f"{column}: {missing_percent:.1f}%")

    validation_results: list[dict[str, Any]] = []

    for feature_set_name, features in feature_sets.items():
        print("\n" + "-" * 70)
        print(f"Training: {feature_set_name}")
        print(f"Feature count: {len(features)}")

        pipeline = build_random_forest_pipeline(
            dataframe=train_df,
            features=features,
            categorical_features=(context_categorical_columns),
        )

        pipeline.fit(
            train_df[features],
            train_df[target],
        )

        validation_predictions = pipeline.predict(validation_df[features])

        metrics = calculate_metrics(
            validation_df[target],
            validation_predictions,
        )

        validation_results.append(
            {
                "model": feature_set_name,
                "feature_count": len(features),
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_r2": metrics["r2"],
            }
        )

        print(f"MAE:  {metrics['mae']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"R²:   {metrics['r2']:.3f}")

    validation_results_df = (
        pd.DataFrame(validation_results)
        .sort_values("validation_mae")
        .reset_index(drop=True)
    )

    print("\nValidation results")
    print(validation_results_df.to_string(index=False))

    safe_to_csv(
        validation_results_df,
        results_file,
    )

    best_feature_set_name = str(
        validation_results_df.loc[
            0,
            "model",
        ]
    )

    best_features = feature_sets[best_feature_set_name]

    print(f"\nSelected model: {best_feature_set_name}")

    final_training_df = model_df.loc[model_df["season"].lt(test_season)].copy()

    final_model = build_random_forest_pipeline(
        dataframe=final_training_df,
        features=best_features,
        categorical_features=(context_categorical_columns),
    )

    final_model.fit(
        final_training_df[best_features],
        final_training_df[target],
    )

    test_predictions = final_model.predict(test_df[best_features])

    test_metrics = calculate_metrics(
        test_df[target],
        test_predictions,
    )

    print("\nFinal test results")
    print(f"MAE:  {test_metrics['mae']:.3f}")
    print(f"RMSE: {test_metrics['rmse']:.3f}")
    print(f"R²:   {test_metrics['r2']:.3f}")

    prediction_columns = [
        column for column in identifier_columns if column in test_df.columns
    ]

    predictions_df = test_df[prediction_columns + [target]].copy()

    predictions_df["predicted_fantasy_points"] = test_predictions

    predictions_df["prediction_error"] = (
        predictions_df[target] - predictions_df["predicted_fantasy_points"]
    )

    predictions_df["absolute_error"] = predictions_df["prediction_error"].abs()

    safe_to_csv(
        predictions_df,
        predictions_file,
    )

    model_bundle = {
        "model": final_model,
        "position": position_value,
        "target": target,
        "features": best_features,
        "feature_set": best_feature_set_name,
        "validation_season": validation_season,
        "test_season": test_season,
        "test_metrics": test_metrics,
    }

    joblib.dump(
        model_bundle,
        model_file,
    )

    print(f"Saved model: {model_file}")
    print("=" * 70)
    print(f"{model_name.upper()} MODEL FINISHED")
    print("=" * 70)
