from pathlib import Path
import pandas as pd

# =====================================================
# Load Dataset
# =====================================================

DATA_DIR = Path(__file__).resolve().parent / "data"

main_df = pd.read_csv(DATA_DIR / "main_df_with_sleeper_ids.csv", low_memory=False)

# =====================================================
# Filter to Quarterbacks
# =====================================================

qb_df = main_df[main_df["position_player_stats"] == "QB"].copy()

# Make sure season/week are numeric
qb_df["season"] = pd.to_numeric(qb_df["season"], errors="coerce")
qb_df["week"] = pd.to_numeric(qb_df["week"], errors="coerce")

# Sort chronologically by player
qb_df = qb_df.sort_values(by=["player_id", "season", "week"])

# =====================================================
# Helper Functions
# =====================================================


def add_last_game_feature(df, column):
    """Previous game value. Carries across seasons."""
    df[f"{column}_last_game"] = df.groupby("player_id")[column].shift(1)
    return df


def add_rolling_average(df, column, window):
    """Rolling average using previous games only. Carries across seasons."""
    df[f"{column}_last_{window}_avg"] = df.groupby("player_id")[column].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    return df


def add_current_season_average(df, column):
    """Season-to-date average using previous games only. Resets every season."""
    df[f"{column}_season_avg"] = df.groupby(["player_id", "season"])[column].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    return df


def add_career_average(df, column):
    """Career average using only games before the current game."""
    df[f"{column}_career_avg"] = df.groupby("player_id")[column].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    return df


def add_ewma_feature(df, column, span=3):
    """Exponentially weighted average using previous games only."""
    df[f"{column}_ewma_{span}"] = df.groupby("player_id")[column].transform(
        lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
    )
    return df


# =====================================================
# Last Game Features
# =====================================================

qb_df = add_last_game_feature(qb_df, "passing_yards")
qb_df = add_last_game_feature(qb_df, "fantasy_points")

# =====================================================
# Rolling Average Features
# These carry across seasons.
# Example: 2026 Week 1 can use end-of-2025 games.
# =====================================================

qb_df = add_rolling_average(qb_df, "passing_yards", 3)
qb_df = add_rolling_average(qb_df, "passing_tds", 3)
qb_df = add_rolling_average(qb_df, "fantasy_points", 5)
qb_df = add_rolling_average(qb_df, "rushing_yards", 3)
qb_df = add_rolling_average(qb_df, "attempts", 3)

# =====================================================
# Current Season Averages
# These reset every season.
# Example: 2026 Week 1 will be NaN because no 2026 games happened yet.
# =====================================================

qb_df = add_current_season_average(qb_df, "fantasy_points")
qb_df = add_current_season_average(qb_df, "passing_yards")
qb_df = add_current_season_average(qb_df, "passing_tds")
qb_df = add_current_season_average(qb_df, "rushing_yards")
qb_df = add_current_season_average(qb_df, "attempts")

# =====================================================
# Career Averages
# These use all previous games before the current game.
# =====================================================

qb_df = add_career_average(qb_df, "fantasy_points")
qb_df = add_career_average(qb_df, "passing_yards")
qb_df = add_career_average(qb_df, "passing_tds")
qb_df = add_career_average(qb_df, "rushing_yards")
qb_df = add_career_average(qb_df, "attempts")

# =====================================================
# EWMA Features
# Recent games are weighted more heavily.
# =====================================================

qb_df = add_ewma_feature(qb_df, "fantasy_points", span=3)
qb_df = add_ewma_feature(qb_df, "passing_yards", span=3)
qb_df = add_ewma_feature(qb_df, "passing_tds", span=3)
qb_df = add_ewma_feature(qb_df, "rushing_yards", span=3)

# =====================================================
# Previous Season Averages
# Example: 2026 rows get the player's 2025 averages.
# =====================================================

previous_season_stats = (
    qb_df.groupby(["player_id", "season"])
    .agg(
        fantasy_points_prev_season=("fantasy_points", "mean"),
        passing_yards_prev_season=("passing_yards", "mean"),
        passing_tds_prev_season=("passing_tds", "mean"),
        rushing_yards_prev_season=("rushing_yards", "mean"),
        attempts_prev_season=("attempts", "mean"),
    )
    .reset_index()
)

previous_season_stats["season"] = previous_season_stats["season"] + 1

qb_df = qb_df.merge(
    previous_season_stats,
    on=["player_id", "season"],
    how="left",
)

# =====================================================
# Save QB Feature Dataset
# =====================================================

output_path = DATA_DIR / "qb_features.csv"
qb_df.to_csv(output_path, index=False)

print(f"QB feature dataset saved to: {output_path}")
print(f"Rows: {len(qb_df)}")
print(
    qb_df[
        [
            "player_display_name",
            "season",
            "week",
            "fantasy_points",
            "fantasy_points_last_game",
            "fantasy_points_last_5_avg",
            "fantasy_points_season_avg",
            "fantasy_points_career_avg",
            "fantasy_points_prev_season",
            "fantasy_points_ewma_3",
        ]
    ].tail(20)
)


# =====================================================
# Feature Selection for QB Model
# =====================================================

TARGET_COLUMN = "fantasy_points"

FEATURE_COLUMNS = [
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
    # Player info
    "years",
    "draft_round",
    "draft_overall",
    "height",
    "weight",
]

# Only keep columns that actually exist
FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col in qb_df.columns]

qb_model_df = qb_df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

# Drop rows where the target is missing
qb_model_df = qb_model_df.dropna(subset=[TARGET_COLUMN])

# Save cleaned modeling dataset
output_path = DATA_DIR / "qb_model_dataset.csv"
qb_model_df.to_csv(output_path, index=False)

print(f"QB model dataset saved to: {output_path}")
print(f"Rows: {len(qb_model_df)}")
print(f"Features used: {len(FEATURE_COLUMNS)}")
print(qb_model_df.head())

qb_df[["weight", "height"]].isna().sum()
len(qb_df)

qb_model_df.isna().sum()
