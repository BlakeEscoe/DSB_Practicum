from pathlib import Path
import pandas as pd
from sleeper_connect import nfl_player_ids


def load_data():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "main_df_with_sleeper_ids.csv"

    return pd.read_csv(data_path)


"""
if __name__ == "__main__":
    df = load_data()

    print(df.head())
    print(df.info())
    print(df.columns.tolist())
"""

# load the jawn
project_root = Path.cwd().parents[0]
data_path = project_root / "Capstone" / "data" / "main_df_with_sleeper_ids.csv"

df = pd.read_csv(data_path)


unq_null_sleeper_df = df[df["sleeper_id"].isnull()].drop_duplicates(subset="player_id")


df_clean = df.dropna(subset=["position_player_stats"])
