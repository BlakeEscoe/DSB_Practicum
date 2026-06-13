"""Tests for the local nflreadpy UI helpers."""

import polars as pl

from nflreadpy.ui import SearchConfig, answer_query, infer_stat, project_player_stat


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
