from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =====================================================
# File Paths
# =====================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "qb_model_dataset.csv"
MODEL_FILE = MODEL_DIR / "qb_random_forest.pkl"
PREDICTIONS_FILE = DATA_DIR / "qb_predictions_test.csv"
IMPORTANCE_FILE = DATA_DIR / "qb_feature_importance.csv"
EXPERIMENT_FILE = DATA_DIR / "qb_model_experiments.csv"


# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Loaded QB model dataset")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# =====================================================
# Target / Debug Columns
# =====================================================

TARGET_COLUMN = "fantasy_points"

DEBUG_COLUMNS = [
    "player_id",
    "player_display_name",
]

DEBUG_COLUMNS = [col for col in DEBUG_COLUMNS if col in df.columns]


# =====================================================
# Train/Test Split
# =====================================================

train_df = df[(df["season"] >= 2001) & (df["season"] <= 2023)].copy()

test_df = df[df["season"] >= 2024].copy()

print("\nTrain rows:", len(train_df))
print("Test rows:", len(test_df))


# =====================================================
# Feature Groups
# =====================================================

BASELINE_FEATURES = [
    # Recent performance
    "passing_yards_last_game",
    "fantasy_points_last_game",
    "passing_yards_last_3_avg",
    "passing_tds_last_3_avg",
    "fantasy_points_last_5_avg",
    "rushing_yards_last_3_avg",
    "attempts_last_3_avg",
    # Current season averages
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
    # Previous season averages
    "fantasy_points_prev_season",
    "passing_yards_prev_season",
    "passing_tds_prev_season",
    "rushing_yards_prev_season",
    "attempts_prev_season",
    # EWMA
    "fantasy_points_ewma_3",
    "passing_yards_ewma_3",
    "passing_tds_ewma_3",
    "rushing_yards_ewma_3",
    # Rookie / history flags
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
    # Missing indicators for context
    "temp_player_stats_missing",
    "wind_player_stats_missing",
    "spread_line_player_stats_missing",
    "total_line_player_stats_missing",
]


DEFENSE_FEATURES = [
    "opp_qb_fantasy_points_allowed_last_4_avg",
    "opp_qb_fantasy_points_allowed_season_avg",
    "opp_qb_fantasy_points_allowed_ewma_4",
    "opp_qb_fantasy_points_allowed_prev_season",
    "opp_qb_passing_yards_allowed_prev_season",
    "opp_qb_rushing_yards_allowed_prev_season",
    "opp_qb_attempts_allowed_prev_season",
]


BETTING_DERIVED_FEATURES = [
    "is_home_team",
    "home_implied_total",
    "away_implied_total",
    "team_implied_total",
    "opponent_implied_total",
    "spread_abs",
]


ROOF_SURFACE_MISSING_FEATURES = [
    "roof_player_stats_missing",
    "surface_player_stats_missing",
]


def keep_existing(cols):
    return [col for col in cols if col in df.columns]


feature_sets = {
    "baseline": keep_existing(BASELINE_FEATURES),
    "baseline_plus_defense": keep_existing(BASELINE_FEATURES + DEFENSE_FEATURES),
    "baseline_plus_betting": keep_existing(
        BASELINE_FEATURES + BETTING_DERIVED_FEATURES
    ),
    "baseline_plus_roof_surface_missing": keep_existing(
        BASELINE_FEATURES + ROOF_SURFACE_MISSING_FEATURES
    ),
    "all_features": keep_existing(
        BASELINE_FEATURES
        + DEFENSE_FEATURES
        + BETTING_DERIVED_FEATURES
        + ROOF_SURFACE_MISSING_FEATURES
    ),
}


# =====================================================
# Model Builder
# =====================================================


def build_model(X_train):
    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_features = [
        col for col in X_train.columns if col not in categorical_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


# =====================================================
# Run Experiments
# =====================================================

experiment_results = []
trained_models = {}

y_train = train_df[TARGET_COLUMN]
y_test = test_df[TARGET_COLUMN]

print("\nRunning feature set experiments...")

for feature_set_name, features in feature_sets.items():
    print(f"\nTraining feature set: {feature_set_name}")
    print(f"Feature count: {len(features)}")

    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    model = build_model(X_train)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    experiment_results.append(
        {
            "feature_set": feature_set_name,
            "feature_count": len(features),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }
    )

    trained_models[feature_set_name] = {
        "model": model,
        "features": features,
        "preds": preds,
    }

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")


# =====================================================
# Pick Best Model by MAE
# =====================================================

experiment_df = pd.DataFrame(experiment_results).sort_values("mae")
experiment_df.to_csv(EXPERIMENT_FILE, index=False)

best_row = experiment_df.iloc[0]
best_feature_set = best_row["feature_set"]

best_model = trained_models[best_feature_set]["model"]
best_features = trained_models[best_feature_set]["features"]
best_preds = trained_models[best_feature_set]["preds"]

print("\n==============================")
print("EXPERIMENT RESULTS")
print("==============================")
print(experiment_df)

print("\n==============================")
print("BEST MODEL SELECTED")
print("==============================")
print(f"Feature set: {best_feature_set}")
print(f"Feature count: {len(best_features)}")
print(f"MAE:  {best_row['mae']:.2f}")
print(f"RMSE: {best_row['rmse']:.2f}")
print(f"R²:   {best_row['r2']:.3f}")
print("==============================")


# =====================================================
# Baseline Comparisons
# =====================================================

X_test_best = test_df[best_features].copy()

baseline_predictions = {
    "Train Mean": [y_train.mean()] * len(y_test),
}

if "fantasy_points_last_game" in X_test_best.columns:
    baseline_predictions["Last Game"] = X_test_best["fantasy_points_last_game"]

if "fantasy_points_last_5_avg" in X_test_best.columns:
    baseline_predictions["Last 5 Avg"] = X_test_best["fantasy_points_last_5_avg"]

if "fantasy_points_ewma_3" in X_test_best.columns:
    baseline_predictions["EWMA 3"] = X_test_best["fantasy_points_ewma_3"]

if "fantasy_points_career_avg" in X_test_best.columns:
    baseline_predictions["Career Avg"] = X_test_best["fantasy_points_career_avg"]

if "fantasy_points_prev_season" in X_test_best.columns:
    baseline_predictions["Previous Season Avg"] = X_test_best[
        "fantasy_points_prev_season"
    ]

print("\nBaseline Comparison")
print("-------------------")

for name, baseline_pred in baseline_predictions.items():
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5
    baseline_r2 = r2_score(y_test, baseline_pred)

    print(f"{name}")
    print(f"  MAE:  {baseline_mae:.2f}")
    print(f"  RMSE: {baseline_rmse:.2f}")
    print(f"  R²:   {baseline_r2:.3f}")


# =====================================================
# Save Best Predictions
# =====================================================

results_cols = DEBUG_COLUMNS + [
    "season",
    "week",
    "team_player_stats",
    "opponent_team",
]

results_cols = [col for col in results_cols if col in test_df.columns]

results_df = test_df[results_cols].copy()
results_df["actual_fantasy_points"] = y_test.values
results_df["predicted_fantasy_points"] = best_preds
results_df["prediction_error"] = (
    results_df["actual_fantasy_points"] - results_df["predicted_fantasy_points"]
)
results_df["absolute_error"] = results_df["prediction_error"].abs()

sort_cols = ["season", "week"]

if "player_display_name" in results_df.columns:
    sort_cols.append("player_display_name")

results_df = results_df.sort_values(by=sort_cols)

results_df.to_csv(PREDICTIONS_FILE, index=False)

print(f"\nSaved best test predictions to: {PREDICTIONS_FILE}")


# =====================================================
# Save Best Model
# =====================================================

joblib.dump(
    {
        "model": best_model,
        "features": best_features,
        "feature_set": best_feature_set,
        "metrics": {
            "mae": best_row["mae"],
            "rmse": best_row["rmse"],
            "r2": best_row["r2"],
        },
    },
    MODEL_FILE,
)

print(f"Saved best model to: {MODEL_FILE}")


# =====================================================
# Feature Importance for Best Model
# =====================================================

rf = best_model.named_steps["random_forest"]
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }
).sort_values("importance", ascending=False)

importance_df.to_csv(IMPORTANCE_FILE, index=False)

print(f"\nSaved feature importance to: {IMPORTANCE_FILE}")

print("\nTop 25 most important features:")
print(importance_df.head(25))


# =====================================================
# Best / Worst Predictions
# =====================================================

print("\nBest 10 predictions:")
print(results_df.sort_values("absolute_error").head(10))

print("\nWorst 10 predictions:")
print(results_df.sort_values("absolute_error", ascending=False).head(10))
