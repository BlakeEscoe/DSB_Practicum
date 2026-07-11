"""Tests for the local nflreadpy UI helpers."""

import polars as pl

from nflreadpy.ui import (
    SearchConfig,
    answer_query,
    get_autocomplete_suggestions,
    infer_stat,
    project_player_stat,
    summarize_split,
)


def sample_stats() -> pl.DataFrame:
    """Build a small stats frame with nflverse-like columns."""
    return pl.DataFrame(
        [
            {
                "player_display_name": "Justin Jefferson",
                "season": 2024,
                "week": 1,
                "recent_team": "MIN",
                "opponent_team": "NYG",
                "receiving_yards": 59,
                "receptions": 4,
                "rushing_yards": 0,
                "passing_yards": 0,
                "fantasy_points": 9.9,
            },
            {
                "player_display_name": "Justin Jefferson",
                "season": 2024,
                "week": 2,
                "recent_team": "MIN",
                "opponent_team": "SF",
                "receiving_yards": 133,
                "receptions": 4,
                "rushing_yards": 0,
                "passing_yards": 0,
                "fantasy_points": 21.3,
            },
            {
                "player_display_name": "Justin Jefferson",
                "season": 2024,
                "week": 3,
                "recent_team": "MIN",
                "opponent_team": "HOU",
                "receiving_yards": 81,
                "receptions": 6,
                "rushing_yards": 0,
                "passing_yards": 0,
                "fantasy_points": 14.1,
            },
            {
                "player_display_name": "Saquon Barkley",
                "season": 2024,
                "week": 1,
                "recent_team": "PHI",
                "opponent_team": "GB",
                "receiving_yards": 23,
                "receptions": 2,
                "rushing_yards": 109,
                "passing_yards": 0,
                "fantasy_points": 32.2,
            },
        ]
    )


def test_infer_stat_from_query_alias() -> None:
    assert (
        infer_stat("Jefferson receiving yards", sample_stats().columns)
        == "receiving_yards"
    )


def test_answer_query_returns_player_projection() -> None:
    answer = answer_query(
        sample_stats(),
        SearchConfig(query="Justin Jefferson receiving yards", seasons=(2024,)),
    )

    assert answer["type"] == "player"
    assert answer["title"] == "Justin Jefferson"
    assert answer["projection"]["stat"] == "receiving_yards"
    assert len(answer["rows"]) == 3


def test_get_autocomplete_suggestions_matches_first_or_last_name() -> None:
    suggestions = get_autocomplete_suggestions(sample_stats(), "jeff", limit=5)

    assert "Justin Jefferson" in suggestions

    team_suggestions = get_autocomplete_suggestions(sample_stats(), "vik", limit=5)
    assert "Vikings" in team_suggestions or "MIN" in team_suggestions


def test_answer_query_returns_leaderboard_without_player_match() -> None:
    answer = answer_query(
        sample_stats(),
        SearchConfig(query="rushing yards leaders", seasons=(2024,)),
    )

    assert answer["type"] == "leaderboard"
    assert answer["rows"][0]["player_display_name"] == "Saquon Barkley"
    assert answer["rows"][0]["rushing_yards"] == 109


def test_project_player_stat_uses_recent_games() -> None:
    projection = project_player_stat(
        sample_stats().filter(pl.col("player_display_name") == "Justin Jefferson"),
        "receiving_yards",
    )

    assert projection is not None
    assert projection["projection"] == 91.0
    assert projection["sample_size"] == 3


def test_answer_query_returns_split_summary_by_day() -> None:
    df = sample_stats().with_columns(
        pl.lit("2024-09-08").alias("game_date"),
        pl.lit("open").alias("roof"),
        pl.lit("Clear").alias("weather"),
        pl.lit(0).alias("div_game"),
        pl.lit(True).alias("is_home"),
    )
    answer = answer_query(
        df,
        SearchConfig(query="Justin Jefferson", seasons=(2024,), split_category="day"),
    )

    assert answer["type"] == "split_summary"
    assert any(row["split"] == "Sunday" for row in answer["rows"])
    # All three sample games share the same (Sunday) game_date, so they
    # collapse into one group and the split sums receiving_yards across them:
    # 59 + 133 + 81.
    assert answer["rows"][0]["receiving_yards"] == 273


def test_answer_query_returns_split_summary_by_group() -> None:
    df = sample_stats().with_columns(
        pl.lit("2024-09-08").alias("game_date"),
        pl.lit("open").alias("roof"),
        pl.lit("Clear").alias("weather"),
        pl.lit(0).alias("div_game"),
        pl.lit(True).alias("is_home"),
        pl.lit("MIN").alias("home_team"),
        pl.lit("NYG").alias("away_team"),
        pl.lit(24).alias("home_score"),
        pl.lit(20).alias("away_score"),
        pl.lit("NFC").alias("home_conf"),
        pl.lit("NFC").alias("away_conf"),
    )
    answer = answer_query(
        df,
        SearchConfig(query="Justin Jefferson", seasons=(2024,), split_category="group"),
    )

    assert answer["type"] == "split_summary"
    assert any("vs NFC" in str(row["split"]) for row in answer["rows"])


def test_answer_query_returns_split_summary_by_outcome() -> None:
    df = sample_stats().with_columns(
        pl.lit("2024-09-08").alias("game_date"),
        pl.lit("open").alias("roof"),
        pl.lit("Clear").alias("weather"),
        pl.lit(0).alias("div_game"),
        pl.lit(True).alias("is_home"),
        pl.lit("MIN").alias("home_team"),
        pl.lit("NYG").alias("away_team"),
        pl.lit(24).alias("home_score"),
        pl.lit(20).alias("away_score"),
        pl.lit("NFC").alias("home_conf"),
        pl.lit("NFC").alias("away_conf"),
    )
    answer = answer_query(
        df,
        SearchConfig(query="Justin Jefferson", seasons=(2024,), split_category="outcome"),
    )

    assert answer["type"] == "split_summary"
    assert any(row["split"] == "Wins/Ties" for row in answer["rows"])
