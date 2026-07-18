from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_predictions(season: int) -> pd.DataFrame:
    """
    Load and combine all fantasy football predictions
    for a requested season.

    Parameters
    ----------
    season : int
        NFL season to load, such as 2025.

    Returns
    -------
    pandas.DataFrame
        Combined QB, RB, WR, TE, K, and DEF predictions.
    """

    prediction_files = {
        "QB": DATA_DIR / f"qb_random_forest_{season}_predictions.csv",
        "RB": DATA_DIR / f"rb_{season}_predictions.csv",
        "WR": DATA_DIR / f"wr_{season}_predictions.csv",
        "TE": DATA_DIR / f"te_{season}_predictions.csv",
        "K": DATA_DIR / f"k_{season}_predictions.csv",
        "DEF": DATA_DIR / f"def_{season}_predictions.csv",
    }

    required_columns = [
        "player_name",
        "season",
        "week",
        "predicted_fantasy_points",
        "position",
    ]

    prediction_frames = []

    for expected_position, file_path in prediction_files.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Prediction file not found for {expected_position}: "
                f"{file_path}"
            )

        position_df = pd.read_csv(file_path)

        missing_columns = [
            column
            for column in required_columns
            if column not in position_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing columns: "
                f"{missing_columns}"
            )

        position_df = position_df[required_columns].copy()

        position_df["season"] = pd.to_numeric(
            position_df["season"],
            errors="coerce",
        )

        position_df["week"] = pd.to_numeric(
            position_df["week"],
            errors="coerce",
        )

        position_df["predicted_fantasy_points"] = pd.to_numeric(
            position_df["predicted_fantasy_points"],
            errors="coerce",
        )

        position_df["position"] = (
            position_df["position"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        position_df = position_df[
            position_df["season"] == season
        ].copy()

        prediction_frames.append(position_df)

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions = predictions.dropna(
        subset=required_columns
    ).copy()

    predictions["season"] = predictions["season"].astype(int)
    predictions["week"] = predictions["week"].astype(int)

    duplicate_mask = predictions.duplicated(
        subset=[
            "player_name",
            "season",
            "week",
            "position",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = predictions.loc[
            duplicate_mask,
            [
                "player_name",
                "season",
                "week",
                "position",
            ],
        ]

        raise ValueError(
            "Duplicate prediction rows detected:\n"
            + duplicate_rows.to_string(index=False)
        )

    predictions = predictions.sort_values(
        [
            "season",
            "week",
            "position",
            "predicted_fantasy_points",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    return predictions


if __name__ == "__main__":

    predictions_2025 = load_predictions(2025)

    print(predictions_2025.head())
    print(
        f"\nTotal prediction rows: "
        f"{len(predictions_2025):,}"
    )

    output_path = DATA_DIR / "all_2025_predictions.csv"

    predictions_2025.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved combined predictions: {output_path}")