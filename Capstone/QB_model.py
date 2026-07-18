from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


# =====================================================
# File Paths
# =====================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "main_df_with_sleeper_ids.csv"
QB_FEATURE_FILE = DATA_DIR / "qb_features.csv"
QB_MODEL_FILE = DATA_DIR / "qb_model_dataset.csv"


# =====================================================
# Safe Save Function
# =====================================================


def safe_to_csv(df, path):
    try:
        df.to_csv(path, index=False)
        print(f"Saved file: {path}")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.stem}_{timestamp}.csv")
        df.to_csv(backup_path, index=False)
        print(f"Permission denied for: {path}")
        print(f"Saved backup instead: {backup_path}")


# =====================================================
# Load Dataset
# =====================================================

main_df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Loaded main dataset")
print("Rows:", len(main_df))

# =====================================================
# Clean Dataset
# =====================================================
game_id_parts = main_df["game_id"].str.split("_", n=3, expand=True)
game_id_parts.columns = ["game_year", "game_week", "game_away_team", "game_home_team"]
game_year = pd.to_numeric(game_id_parts["game_year"], errors="coerce")
cleanup_year = game_year.combine_first(
    pd.to_numeric(main_df["season"], errors="coerce")
)


def normalize_team_abbreviations(team_series, year_series):
    # Team abbreviation cleanup for historical franchise moves.
    cleaned_team_series = team_series.replace({"JAC": "JAX"})

    cleaned_team_series.loc[year_series < 2020] = cleaned_team_series.loc[
        year_series < 2020
    ].replace({"LV": "OAK"})
    cleaned_team_series.loc[year_series >= 2020] = cleaned_team_series.loc[
        year_series >= 2020
    ].replace({"OAK": "LV"})

    cleaned_team_series.loc[year_series < 2017] = cleaned_team_series.loc[
        year_series < 2017
    ].replace({"LAC": "SD"})
    cleaned_team_series.loc[year_series >= 2017] = cleaned_team_series.loc[
        year_series >= 2017
    ].replace({"SD": "LAC"})

    cleaned_team_series.loc[year_series < 2016] = cleaned_team_series.loc[
        year_series < 2016
    ].replace({"LA": "STL"})
    cleaned_team_series.loc[year_series >= 2016] = cleaned_team_series.loc[
        year_series >= 2016
    ].replace({"STL": "LA"})

    return cleaned_team_series


for team_col in ["game_away_team", "game_home_team"]:
    game_id_parts[team_col] = normalize_team_abbreviations(
        game_id_parts[team_col], cleanup_year
    )

main_df["away_team_player_stats"] = normalize_team_abbreviations(
    main_df["away_team_player_stats"], cleanup_year
)
main_df["home_team_player_stats"] = normalize_team_abbreviations(
    main_df["home_team_player_stats"], cleanup_year
)
main_df["team_player_stats"] = normalize_team_abbreviations(
    main_df["team_player_stats"], cleanup_year
)
main_df["opponent_team"] = normalize_team_abbreviations(
    main_df["opponent_team"], cleanup_year
)

has_game_id = main_df["game_id"].notna()
main_df.loc[has_game_id, "game_id"] = (
    game_id_parts.loc[has_game_id, "game_year"]
    + "_"
    + game_id_parts.loc[has_game_id, "game_week"]
    + "_"
    + game_id_parts.loc[has_game_id, "game_away_team"]
    + "_"
    + game_id_parts.loc[has_game_id, "game_home_team"]
)
main_df.loc[has_game_id, "away_team_player_stats"] = game_id_parts.loc[
    has_game_id, "game_away_team"
]
main_df.loc[has_game_id, "home_team_player_stats"] = game_id_parts.loc[
    has_game_id, "game_home_team"
]

main_df["opponent_team"] = np.select(
    [
        main_df["team_player_stats"] == main_df["away_team_player_stats"],
        main_df["team_player_stats"] == main_df["home_team_player_stats"],
    ],
    [
        main_df["home_team_player_stats"],
        main_df["away_team_player_stats"],
    ],
    default=main_df["opponent_team"],
)

main_df[
    ["game_id", "team_player_stats", "opponent_team", "home_team_player_stats"]
].isna().sum()
main_df[main_df["game_id"].isna()][
    [
        "game_id",
        "season",
        "week",
        "team_player_stats",
        "opponent_team",
        "away_team_player_stats",
        "home_team_player_stats",
    ]
].head(20)

# =====================================================
# Filter to Quarterbacks
# =====================================================

qb_df = main_df[main_df["position_player_stats"] == "QB"].copy()
qb_df = qb_df.dropna(subset=["player_id", "fantasy_points"]).copy()
qb_df["player_id"] = qb_df["player_id"].astype(str).str.strip()

print("\nQB rows after dropping missing player_id:")
print(len(qb_df))

qb_df.head(20)
main_df.head(20)
main_df[
    [
        "game_id",
        "season",
        "week",
        "team_player_stats",
        "opponent_team",
        "wind_player_stats",
        "temp_player_stats",
    ]
].head(20)
main_df[["wind_player_stats", "temp_player_stats"]].isna().sum()
main_df[
    [
        "game_id",
        "season",
        "week",
        "team_player_stats",
        "opponent_team",
        "away_team_player_stats",
        "home_team_player_stats",
    ]
].isna().sum()
main_df[main_df["opponent_team"].isna()][
    [
        "game_id",
        "season",
        "week",
        "team_player_stats",
        "opponent_team",
        "away_team_player_stats",
        "home_team_player_stats",
    ]
]

main_df[main_df["game_id"].isna()][
    [
        "game_id",
        "season",
        "week",
        "team_player_stats",
        "opponent_team",
        "away_team_player_stats",
        "home_team_player_stats",
    ]
].isna().sum()
main_df[main_df["game_id"].isna()][
    [
        "game_id",
        "season",
        "week",
        "team_player_stats",
        "opponent_team",
        "away_team_player_stats",
        "home_team_player_stats",
    ]
].isna().sum()

# =====================================================
# Convert Important Columns to Numeric
# =====================================================

numeric_cols = [
    "season",
    "week",
    "passing_yards",
    "passing_tds",
    "fantasy_points",
    "rushing_yards",
    "attempts",
    "temp_player_stats",
    "wind_player_stats",
    "spread_line_player_stats",
    "total_line_player_stats",
]

for col in numeric_cols:
    if col in qb_df.columns:
        qb_df[col] = pd.to_numeric(qb_df[col], errors="coerce")


# =====================================================
# Sort Chronologically
# =====================================================

qb_df = qb_df.sort_values(by=["player_id", "season", "week"]).copy()


# =====================================================
# Player Feature Engineering Helper Functions
# =====================================================


def add_last_game_feature(df, column):
    df[f"{column}_last_game"] = df.groupby("player_id")[column].shift(1)
    return df


def add_rolling_average(df, column, window):
    df[f"{column}_last_{window}_avg"] = df.groupby("player_id")[column].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    return df


def add_current_season_average(df, column):
    df[f"{column}_season_avg"] = df.groupby(["player_id", "season"])[column].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    return df


def add_career_average(df, column):
    df[f"{column}_career_avg"] = df.groupby("player_id")[column].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    return df


def add_ewma_feature(df, column, span=3):
    df[f"{column}_ewma_{span}"] = df.groupby("player_id")[column].transform(
        lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
    )
    return df


# =====================================================
# Player History Features
# =====================================================

qb_df = add_last_game_feature(qb_df, "passing_yards")
qb_df = add_last_game_feature(qb_df, "fantasy_points")

qb_df = add_rolling_average(qb_df, "passing_yards", 3)
qb_df = add_rolling_average(qb_df, "passing_tds", 3)
qb_df = add_rolling_average(qb_df, "fantasy_points", 5)
qb_df = add_rolling_average(qb_df, "rushing_yards", 3)
qb_df = add_rolling_average(qb_df, "attempts", 3)

qb_df = add_current_season_average(qb_df, "fantasy_points")
qb_df = add_current_season_average(qb_df, "passing_yards")
qb_df = add_current_season_average(qb_df, "passing_tds")
qb_df = add_current_season_average(qb_df, "rushing_yards")
qb_df = add_current_season_average(qb_df, "attempts")

qb_df = add_career_average(qb_df, "fantasy_points")
qb_df = add_career_average(qb_df, "passing_yards")
qb_df = add_career_average(qb_df, "passing_tds")
qb_df = add_career_average(qb_df, "rushing_yards")
qb_df = add_career_average(qb_df, "attempts")

qb_df = add_ewma_feature(qb_df, "fantasy_points", span=3)
qb_df = add_ewma_feature(qb_df, "passing_yards", span=3)
qb_df = add_ewma_feature(qb_df, "passing_tds", span=3)
qb_df = add_ewma_feature(qb_df, "rushing_yards", span=3)


# =====================================================
# Previous Season Player Averages
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
# Opponent Defensive Strength Features
# =====================================================

defense_weekly = (
    qb_df.groupby(["opponent_team", "season", "week"])
    .agg(
        opp_qb_fantasy_points_allowed=("fantasy_points", "sum"),
        opp_qb_passing_yards_allowed=("passing_yards", "sum"),
        opp_qb_rushing_yards_allowed=("rushing_yards", "sum"),
        opp_qb_attempts_allowed=("attempts", "sum"),
    )
    .reset_index()
)

defense_weekly = defense_weekly.rename(columns={"opponent_team": "defense_team"})

defense_weekly = defense_weekly.sort_values(
    by=["defense_team", "season", "week"]
).copy()


def add_defense_rolling_average(df, column, window):
    df[f"{column}_last_{window}_avg"] = df.groupby("defense_team")[column].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    return df


def add_defense_season_average(df, column):
    df[f"{column}_season_avg"] = df.groupby(["defense_team", "season"])[
        column
    ].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    return df


def add_defense_ewma(df, column, span=4):
    df[f"{column}_ewma_{span}"] = df.groupby("defense_team")[column].transform(
        lambda x: x.shift(1).ewm(span=span, adjust=False).mean()
    )
    return df


defense_weekly = add_defense_rolling_average(
    defense_weekly,
    "opp_qb_fantasy_points_allowed",
    window=4,
)

defense_weekly = add_defense_season_average(
    defense_weekly,
    "opp_qb_fantasy_points_allowed",
)

defense_weekly = add_defense_ewma(
    defense_weekly,
    "opp_qb_fantasy_points_allowed",
    span=4,
)


defense_prev_season = (
    defense_weekly.groupby(["defense_team", "season"])
    .agg(
        opp_qb_fantasy_points_allowed_prev_season=(
            "opp_qb_fantasy_points_allowed",
            "mean",
        ),
        opp_qb_passing_yards_allowed_prev_season=(
            "opp_qb_passing_yards_allowed",
            "mean",
        ),
        opp_qb_rushing_yards_allowed_prev_season=(
            "opp_qb_rushing_yards_allowed",
            "mean",
        ),
        opp_qb_attempts_allowed_prev_season=(
            "opp_qb_attempts_allowed",
            "mean",
        ),
    )
    .reset_index()
)

defense_prev_season["season"] = defense_prev_season["season"] + 1

defense_weekly = defense_weekly.merge(
    defense_prev_season,
    on=["defense_team", "season"],
    how="left",
)

defense_feature_cols = [
    "opp_qb_fantasy_points_allowed_last_4_avg",
    "opp_qb_fantasy_points_allowed_season_avg",
    "opp_qb_fantasy_points_allowed_ewma_4",
    "opp_qb_fantasy_points_allowed_prev_season",
    "opp_qb_passing_yards_allowed_prev_season",
    "opp_qb_rushing_yards_allowed_prev_season",
    "opp_qb_attempts_allowed_prev_season",
]

defense_merge_cols = ["defense_team", "season", "week"] + defense_feature_cols

qb_df = qb_df.merge(
    defense_weekly[defense_merge_cols],
    left_on=["opponent_team", "season", "week"],
    right_on=["defense_team", "season", "week"],
    how="left",
)

qb_df = qb_df.drop(columns=["defense_team"])


# =====================================================
# Detect Rookie / First Game Situations
# =====================================================

qb_df = qb_df.sort_values(by=["player_id", "season", "week"]).copy()

qb_df["career_game_number"] = qb_df.groupby("player_id").cumcount() + 1

qb_df["is_first_observed_game"] = (qb_df["career_game_number"] == 1).astype(int)

qb_df["season_game_number"] = qb_df.groupby(["player_id", "season"]).cumcount() + 1

qb_df["is_first_game_of_season"] = (qb_df["season_game_number"] == 1).astype(int)

qb_df["first_observed_season"] = qb_df.groupby("player_id")["season"].transform("min")

DATA_START_SEASON = qb_df["season"].min()

if "draft_year" in qb_df.columns:
    qb_df["draft_year"] = pd.to_numeric(qb_df["draft_year"], errors="coerce")

    qb_df["is_known_rookie_season"] = (qb_df["draft_year"] == qb_df["season"]).astype(
        int
    )
else:
    qb_df["is_known_rookie_season"] = 0

qb_df["is_possible_rookie_season"] = (
    (qb_df["first_observed_season"] == qb_df["season"])
    & (qb_df["season"] > DATA_START_SEASON)
).astype(int)

if "draft_year" in qb_df.columns:
    qb_df["is_rookie_season"] = (
        (qb_df["is_known_rookie_season"] == 1)
        | (qb_df["draft_year"].isna() & (qb_df["is_possible_rookie_season"] == 1))
    ).astype(int)
else:
    qb_df["is_rookie_season"] = qb_df["is_possible_rookie_season"]

qb_df["is_rookie_first_game"] = (
    (qb_df["is_first_observed_game"] == 1) & (qb_df["is_rookie_season"] == 1)
).astype(int)

qb_df["has_prior_qb_game"] = (qb_df["is_first_observed_game"] == 0).astype(int)


# =====================================================
# Fill Engineered Player History Nulls
# =====================================================

history_features = [
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

history_features = [col for col in history_features if col in qb_df.columns]

for col in history_features:
    qb_df.loc[
        (qb_df["is_rookie_first_game"] == 1) & (qb_df[col].isna()),
        col,
    ] = 0

for col in history_features:
    qb_df[col] = qb_df[col].fillna(0)


# =====================================================
# Fill Current Season Average Nulls
# =====================================================

season_avg_features = [
    "fantasy_points_season_avg",
    "passing_yards_season_avg",
    "passing_tds_season_avg",
    "rushing_yards_season_avg",
    "attempts_season_avg",
]

season_avg_features = [col for col in season_avg_features if col in qb_df.columns]

for col in season_avg_features:
    qb_df.loc[
        (qb_df["is_first_game_of_season"] == 1) & (qb_df[col].isna()),
        col,
    ] = 0

for col in season_avg_features:
    qb_df[col] = qb_df[col].fillna(0)


# =====================================================
# Fill Previous Season Player Average Nulls
# =====================================================

prev_to_career_map = {
    "fantasy_points_prev_season": "fantasy_points_career_avg",
    "passing_yards_prev_season": "passing_yards_career_avg",
    "passing_tds_prev_season": "passing_tds_career_avg",
    "rushing_yards_prev_season": "rushing_yards_career_avg",
    "attempts_prev_season": "attempts_career_avg",
}

for prev_col, career_col in prev_to_career_map.items():
    if prev_col in qb_df.columns:
        qb_df.loc[
            (qb_df["is_rookie_first_game"] == 1) & (qb_df[prev_col].isna()),
            prev_col,
        ] = 0

        if career_col in qb_df.columns:
            qb_df[prev_col] = qb_df[prev_col].fillna(qb_df[career_col])

        qb_df[prev_col] = qb_df[prev_col].fillna(0)


# =====================================================
# Fill Opponent Defensive Feature Nulls
# =====================================================

defense_feature_cols = [
    "opp_qb_fantasy_points_allowed_last_4_avg",
    "opp_qb_fantasy_points_allowed_season_avg",
    "opp_qb_fantasy_points_allowed_ewma_4",
    "opp_qb_fantasy_points_allowed_prev_season",
    "opp_qb_passing_yards_allowed_prev_season",
    "opp_qb_rushing_yards_allowed_prev_season",
    "opp_qb_attempts_allowed_prev_season",
]

defense_feature_cols = [col for col in defense_feature_cols if col in qb_df.columns]

for col in defense_feature_cols:
    qb_df[col] = qb_df[col].fillna(
        qb_df.groupby(["season", "week"])[col].transform("median")
    )

    median_value = qb_df[col].median()

    if pd.isna(median_value):
        median_value = 0

    qb_df[col] = qb_df[col].fillna(median_value)


# =====================================================
# Basic Context Cleanup
# =====================================================

for col in ["team_player_stats", "opponent_team"]:
    if col in qb_df.columns:
        qb_df[col] = qb_df[col].fillna("Unknown").astype(str)


# =====================================================
# Create Venue Key
# =====================================================

stadium_candidates = [
    "stadium_player_stats",
    "stadium",
    "stadium_name",
    "stadium_id_player_stats",
]

stadium_col = next(
    (col for col in stadium_candidates if col in qb_df.columns),
    None,
)

if stadium_col is not None:
    qb_df["venue_key"] = qb_df[stadium_col].fillna("Unknown").astype(str)
    print(f"Using {stadium_col} for venue/weather imputation.")

elif "home_team_player_stats" in qb_df.columns:
    qb_df["venue_key"] = qb_df["home_team_player_stats"].fillna("Unknown").astype(str)
    print("Using home_team_player_stats as venue proxy.")

else:
    qb_df["venue_key"] = "Unknown"
    print("No stadium/home team column found. Using league averages.")


# =====================================================
# Impute Roof / Surface From Venue History
# =====================================================


def fill_categorical_from_venue_history(df, col):
    if col not in df.columns:
        return df

    missing_col = f"{col}_missing"

    df[col] = df[col].replace(["", "nan", "None", "Unknown"], pd.NA)
    df[missing_col] = df[col].isna().astype(int)

    def mode_or_none(series):
        series = series.dropna()

        if len(series) == 0:
            return None

        return series.mode().iloc[0]

    missing_idx = df.index[df[col].isna()]

    for idx in missing_idx:
        row = df.loc[idx]

        season = row["season"]
        venue = row["venue_key"]

        home_team = None
        if "home_team_player_stats" in df.columns:
            home_team = row["home_team_player_stats"]

        fill_value = None

        if venue != "Unknown":
            values = df.loc[
                (df["venue_key"] == venue) & (df["season"] == season),
                col,
            ]
            fill_value = mode_or_none(values)

        if fill_value is None and venue != "Unknown":
            values = df.loc[
                df["venue_key"] == venue,
                col,
            ]
            fill_value = mode_or_none(values)

        if (
            fill_value is None
            and home_team is not None
            and "home_team_player_stats" in df.columns
        ):
            values = df.loc[
                (df["home_team_player_stats"] == home_team) & (df["season"] == season),
                col,
            ]
            fill_value = mode_or_none(values)

        if (
            fill_value is None
            and home_team is not None
            and "home_team_player_stats" in df.columns
        ):
            values = df.loc[
                df["home_team_player_stats"] == home_team,
                col,
            ]
            fill_value = mode_or_none(values)

        if fill_value is None:
            fill_value = "Unknown"

        df.at[idx, col] = fill_value

    df[col] = df[col].fillna("Unknown").astype(str)

    return df


qb_df = fill_categorical_from_venue_history(qb_df, "roof_player_stats")
qb_df = fill_categorical_from_venue_history(qb_df, "surface_player_stats")


# =====================================================
# Indoor / Outdoor Logic
# =====================================================

if "roof_player_stats" in qb_df.columns:
    roof_lower = qb_df["roof_player_stats"].astype(str).str.lower()
else:
    roof_lower = pd.Series("", index=qb_df.index)

indoor_mask = roof_lower.str.contains("dome|closed", na=False)


# =====================================================
# Weather Imputation
# =====================================================


def fill_weather_from_venue_history(df, col, indoor_default):
    if col not in df.columns:
        return df

    missing_col = f"{col}_missing"

    df[missing_col] = df[col].isna().astype(int)

    df.loc[
        indoor_mask & df[col].isna(),
        col,
    ] = indoor_default

    observed = df[(df[col].notna()) & (df[missing_col] == 0)].copy()

    def get_mean(values, min_count=1):
        values = values.dropna()

        if len(values) >= min_count:
            return values.mean()

        return None

    missing_outdoor_idx = df.index[(~indoor_mask) & (df[col].isna())]

    for idx in missing_outdoor_idx:
        row = df.loc[idx]

        season = row["season"]
        week = row["week"]
        venue = row["venue_key"]

        fill_value = None

        if venue != "Unknown":
            values = observed.loc[
                (observed["venue_key"] == venue)
                & (observed["season"] == season)
                & (observed["week"] < week)
                & (observed["week"] >= week - 4),
                col,
            ]
            fill_value = get_mean(values, min_count=1)

        if fill_value is None and venue != "Unknown":
            values = observed.loc[
                (observed["venue_key"] == venue)
                & (observed["season"] < season)
                & (observed["week"].between(week - 2, week + 2)),
                col,
            ]
            fill_value = get_mean(values, min_count=2)

        if fill_value is None and venue != "Unknown":
            values = observed.loc[
                (observed["venue_key"] == venue)
                & (observed["season"] < season)
                & (observed["week"].between(week - 4, week + 4)),
                col,
            ]
            fill_value = get_mean(values, min_count=2)

        if fill_value is None:
            values = observed.loc[
                (observed["season"] < season)
                & (observed["week"].between(week - 2, week + 2)),
                col,
            ]
            fill_value = get_mean(values, min_count=5)

        if fill_value is None:
            fill_value = observed[col].median()

        if pd.isna(fill_value):
            fill_value = indoor_default

        df.at[idx, col] = fill_value

    return df


qb_df = fill_weather_from_venue_history(
    qb_df,
    col="temp_player_stats",
    indoor_default=70,
)

qb_df = fill_weather_from_venue_history(
    qb_df,
    col="wind_player_stats",
    indoor_default=0,
)


# =====================================================
# Betting Lines Imputation
# =====================================================

betting_cols = [
    "spread_line_player_stats",
    "total_line_player_stats",
]

for col in betting_cols:
    if col in qb_df.columns:
        qb_df[f"{col}_missing"] = qb_df[col].isna().astype(int)

        qb_df[col] = qb_df[col].fillna(
            qb_df.groupby(["season", "week"])[col].transform("median")
        )

        median_value = qb_df[col].median()

        if pd.isna(median_value):
            median_value = 0

        qb_df[col] = qb_df[col].fillna(median_value)


# =====================================================
# Betting-Derived Features
# =====================================================

needed_betting_cols = [
    "spread_line_player_stats",
    "total_line_player_stats",
    "team_player_stats",
    "home_team_player_stats",
    "away_team_player_stats",
]

if all(col in qb_df.columns for col in needed_betting_cols):
    qb_df["is_home_team"] = (
        qb_df["team_player_stats"] == qb_df["home_team_player_stats"]
    ).astype(int)

    # Assumes spread_line_player_stats is from the home team's perspective.
    # Negative spread means the home team is favored.
    qb_df["home_implied_total"] = (
        qb_df["total_line_player_stats"] / 2 - qb_df["spread_line_player_stats"] / 2
    )

    qb_df["away_implied_total"] = (
        qb_df["total_line_player_stats"] / 2 + qb_df["spread_line_player_stats"] / 2
    )

    qb_df["team_implied_total"] = qb_df["away_implied_total"]

    qb_df.loc[
        qb_df["team_player_stats"] == qb_df["home_team_player_stats"],
        "team_implied_total",
    ] = qb_df["home_implied_total"]

    qb_df["opponent_implied_total"] = qb_df["home_implied_total"]

    qb_df.loc[
        qb_df["team_player_stats"] == qb_df["home_team_player_stats"],
        "opponent_implied_total",
    ] = qb_df["away_implied_total"]

    qb_df["spread_abs"] = qb_df["spread_line_player_stats"].abs()

else:
    qb_df["is_home_team"] = 0
    qb_df["home_implied_total"] = qb_df["total_line_player_stats"] / 2
    qb_df["away_implied_total"] = qb_df["total_line_player_stats"] / 2
    qb_df["team_implied_total"] = qb_df["total_line_player_stats"] / 2
    qb_df["opponent_implied_total"] = qb_df["total_line_player_stats"] / 2
    qb_df["spread_abs"] = qb_df["spread_line_player_stats"].abs()


# =====================================================
# Save Full QB Feature Dataset
# =====================================================

safe_to_csv(qb_df, QB_FEATURE_FILE)

print(f"\nQB feature dataset saved to: {QB_FEATURE_FILE}")
print(f"Rows: {len(qb_df)}")


# =====================================================
# Build Final QB Model Dataset
# =====================================================

TARGET_COLUMN = "fantasy_points"

DEBUG_COLUMNS = [
    "player_id",
    "player_display_name",
]

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
    # Opponent defensive strength
    "opp_qb_fantasy_points_allowed_last_4_avg",
    "opp_qb_fantasy_points_allowed_season_avg",
    "opp_qb_fantasy_points_allowed_ewma_4",
    "opp_qb_fantasy_points_allowed_prev_season",
    "opp_qb_passing_yards_allowed_prev_season",
    "opp_qb_rushing_yards_allowed_prev_season",
    "opp_qb_attempts_allowed_prev_season",
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
    # Betting-derived features
    "is_home_team",
    "home_implied_total",
    "away_implied_total",
    "team_implied_total",
    "opponent_implied_total",
    "spread_abs",
    # Missing indicators for context
    "roof_player_stats_missing",
    "surface_player_stats_missing",
    "temp_player_stats_missing",
    "wind_player_stats_missing",
    "spread_line_player_stats_missing",
    "total_line_player_stats_missing",
]

DEBUG_COLUMNS = [col for col in DEBUG_COLUMNS if col in qb_df.columns]
FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col in qb_df.columns]

qb_model_df = qb_df[DEBUG_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

qb_model_df = qb_model_df.dropna(subset=[TARGET_COLUMN])


# =====================================================
# Remove 2000 Warm-Up Season
# =====================================================

qb_model_df = qb_model_df[qb_model_df["season"] >= 2001].copy()


# =====================================================
# Final Missing Value Check
# =====================================================

missing = qb_model_df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\nMissing values in final QB model dataset:")
if len(missing) == 0:
    print("No missing values.")
else:
    print(missing)


# =====================================================
# Save Final QB Model Dataset
# =====================================================

safe_to_csv(qb_model_df, QB_MODEL_FILE)

print("\nFinal QB model dataset created.")
print("Rows:", len(qb_model_df))
print("Feature count:", len(FEATURE_COLUMNS))
print("Season range:", qb_model_df["season"].min(), "-", qb_model_df["season"].max())

print("\nSample rows:")
sample_cols = DEBUG_COLUMNS + [
    "season",
    "week",
    "team_player_stats",
    "opponent_team",
    "fantasy_points",
    "fantasy_points_last_game",
    "fantasy_points_last_5_avg",
    "fantasy_points_prev_season",
    "opp_qb_fantasy_points_allowed_last_4_avg",
    "team_implied_total",
    "is_rookie_first_game",
    "has_prior_qb_game",
]

sample_cols = [col for col in sample_cols if col in qb_model_df.columns]

print(qb_model_df[sample_cols].head(20))
