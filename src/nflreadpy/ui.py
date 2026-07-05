"""A small local web UI for exploring nflreadpy player stats.

Run it with:

    python -m nflreadpy.ui
"""

from __future__ import annotations

import os
import argparse
import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse
import requests

import polars as pl
import logging
import traceback

from .old_nfl_files.load_stats import load_player_stats
from .old_nfl_files.load_schedules import load_schedules
from .utils_date import get_current_season

STAT_ALIASES: dict[str, str] = {
    "passing": "passing_yards",
    "passing yards": "passing_yards",
    "pass yards": "passing_yards",
    "passing tds": "passing_tds",
    "pass tds": "passing_tds",
    "passing touchdowns": "passing_tds",
    "interceptions": "interceptions",
    "ints": "interceptions",
    "rushing": "rushing_yards",
    "rushing yards": "rushing_yards",
    "rush yards": "rushing_yards",
    "carries": "carries",
    "rushing tds": "rushing_tds",
    "rush tds": "rushing_tds",
    "rushing touchdowns": "rushing_tds",
    "receiving": "receiving_yards",
    "receiving yards": "receiving_yards",
    "receiver yards": "receiving_yards",
    "rec yards": "receiving_yards",
    "receptions": "receptions",
    "catches": "receptions",
    "targets": "targets",
    "receiving tds": "receiving_tds",
    "receiving touchdowns": "receiving_tds",
    "fantasy points": "fantasy_points_ppr",
    "fantasy": "fantasy_points_ppr",
    "ppr": "fantasy_points_ppr",
    "points per reception": "fantasy_points_ppr",
}

TEAM_ALIASES: dict[str, str] = {
    "detroit lions": "DET",
    "lions": "DET",
    "detroit": "DET",
    "minnesota vikings": "MIN",
    "vikings": "MIN",
    "minnesota": "MIN",
    "chicago bears": "CHI",
    "bears": "CHI",
    "chicago": "CHI",
    "green bay packers": "GB",
    "packers": "GB",
    "green bay": "GB",
    "dallas cowboys": "DAL",
    "cowboys": "DAL",
    "dallas": "DAL",
    "new england patriots": "NE",
    "patriots": "NE",
    "new england": "NE",
    "new york giants": "NYG",
    "giants": "NYG",
    "nyg": "NYG",
    "new york jets": "NYJ",
    "jets": "NYJ",
    "nyj": "NYJ",
    "miami dolphins": "MIA",
    "dolphins": "MIA",
    "miami": "MIA",
    "buffalo bills": "BUF",
    "bills": "BUF",
    "buffalo": "BUF",
    "philadelphia eagles": "PHI",
    "eagles": "PHI",
    "philadelphia": "PHI",
    "pittsburgh steelers": "PIT",
    "steelers": "PIT",
    "pittsburgh": "PIT",
    "baltimore ravens": "BAL",
    "ravens": "BAL",
    "baltimore": "BAL",
    "cincinnati bengals": "CIN",
    "bengals": "CIN",
    "cincinnati": "CIN",
    "cleveland browns": "CLE",
    "browns": "CLE",
    "cleveland": "CLE",
    "indianapolis colts": "IND",
    "colts": "IND",
    "indianapolis": "IND",
    "jacksonville jaguars": "JAX",
    "jaguars": "JAX",
    "jacksonville": "JAX",
    "houston texans": "HOU",
    "texans": "HOU",
    "houston": "HOU",
    "tennessee titans": "TEN",
    "titans": "TEN",
    "tennessee": "TEN",
    "atlanta falcons": "ATL",
    "falcons": "ATL",
    "atlanta": "ATL",
    "carolina panthers": "CAR",
    "panthers": "CAR",
    "carolina": "CAR",
    "new orleans saints": "NO",
    "saints": "NO",
    "new orleans": "NO",
    "tampa bay buccaneers": "TB",
    "buccaneers": "TB",
    "tampa bay": "TB",
    "arizona cardinals": "ARI",
    "cardinals": "ARI",
    "arizona": "ARI",
    "san francisco 49ers": "SF",
    "49ers": "SF",
    "niners": "SF",
    "san francisco": "SF",
    "los angeles rams": "LAR",
    "rams": "LAR",
    "los angeles chargers": "LAC",
    "chargers": "LAC",
    "las vegas raiders": "LV",
    "raiders": "LV",
    "las vegas": "LV",
    "denver broncos": "DEN",
    "broncos": "DEN",
    "denver": "DEN",
    "kansas city chiefs": "KC",
    "chiefs": "KC",
    "kansas city": "KC",
}

QUERY_STOP_WORDS = {
    "best",
    "leader",
    "leaders",
    "most",
    "player",
    "players",
    "predict",
    "prediction",
    "project",
    "projection",
    "stat",
    "stats",
    "top",
}

DEFAULT_STATS = [
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "targets",
    "fantasy_points_ppr",
    "fantasy_points",
]

DISPLAY_COLUMNS = [
    "season",
    "week",
    "recent_team",
    "opponent_team",
    "passing_yards",
    "passing_tds",
    "rushing_yards",
    "rushing_tds",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "fantasy_points_ppr",
    "fantasy_points",
]


@dataclass(frozen=True)
class SearchConfig:
    """User-facing query settings."""

    query: str
    seasons: tuple[int, ...]
    limit: int = 8
    compare: tuple[str, ...] = ()


def parse_compare_names(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()

    names = re.split(r"\s*(?:,|vs|and|&)\s*", value.strip(), flags=re.IGNORECASE)
    return tuple(name.strip() for name in names if name.strip())[:3]


def recent_seasons(count: int = 3) -> tuple[int, ...]:
    """Return the recent seasons used by the UI by default."""
    current = get_current_season()
    return tuple(range(current - count + 1, current + 1))


def parse_seasons(value: str | None) -> tuple[int, ...]:
    """Parse a comma-separated season list."""
    if not value:
        return recent_seasons()

    seasons = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        seasons.append(int(part))

    return tuple(sorted(set(seasons))) or recent_seasons()


def infer_stat(query: str, columns: list[str]) -> str | None:
    """Infer the most likely stat column from a natural-language query."""
    normalized = query.lower().replace("_", " ")

    for phrase, column in sorted(
        STAT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if phrase in normalized and column in columns:
            return column

    for column in columns:
        readable = column.replace("_", " ")
        if readable in normalized:
            return column

    return None


def default_stat(columns: list[str]) -> str | None:
    """Pick a reasonable default stat from available columns."""
    for column in DEFAULT_STATS:
        if column in columns:
            return column
    return None


def _query_team_location(query: str) -> str | None:
    normalized = query.lower()
    if "away" in normalized:
        return "away"
    if "home" in normalized and "away" not in normalized:
        return "home"
    return None


def _find_team_alias(query: str) -> str | None:
    normalized = query.lower().strip()
    for alias, code in TEAM_ALIASES.items():
        if alias in normalized:
            return code
        if normalized == code.lower():
            return code
    return None


def suggest_names(df: pl.DataFrame, query: str, limit: int = 20) -> list[str]:
    normalized_query = query.lower().strip()
    if not normalized_query:
        return []

    candidate_columns = [
        col for col in (
            "player_display_name",
            "player_name",
            "display_name",
            "name",
            "recent_team",
            "team",
        )
        if col in df.columns
    ]

    candidates: set[str] = set()
    for col in candidate_columns:
        candidates.update(
            str(value)
            for value in df.select(pl.col(col)).unique().to_series().to_list()
            if value is not None and str(value).strip()
        )

    # Include city/team alias strings and team codes so team search suggestions are available.
    candidates.update(TEAM_ALIASES.keys())
    candidates.update(TEAM_ALIASES.values())

    def format_suggestion(value: str) -> str:
        if value in TEAM_ALIASES.keys():
            return value.title()
        if value in TEAM_ALIASES.values():
            return value.upper()
        return value

    scored: list[tuple[int, int, int, int, int, int, int, str]] = []
    for value in candidates:
        normalized = value.lower().strip()
        if not normalized or normalized_query not in normalized:
            continue

        exact_match = 2 if normalized == normalized_query else 0
        starts_with = 2 if normalized.startswith(normalized_query) else 0
        whole_word = 1 if re.search(rf"\b{re.escape(normalized_query)}\b", normalized) else 0
        word_count = len(normalized.split())
        dot_score = 1 if "." in normalized else 0
        scored.append((exact_match, starts_with, whole_word, word_count, dot_score, -len(normalized), format_suggestion(value)))

    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]), reverse=True)
    return [item[-1] for item in scored][:limit]


def best_player_stat(df: pl.DataFrame) -> str | None:
    """Pick the most informative default stat for a matched player."""
    candidates = [column for column in DEFAULT_STATS if column in df.columns]
    if not candidates:
        return None

    totals = df.select(
        [pl.col(column).sum().alias(column) for column in candidates]
    ).to_dicts()[0]
    return max(candidates, key=lambda column: float(totals.get(column) or 0))


def _contains_words_expr(column: str, words: list[str]) -> pl.Expr:
    expr = pl.lit(True)
    lowered = pl.col(column).cast(pl.Utf8).str.to_lowercase()
    for word in words:
        expr = expr & lowered.str.contains(word, literal=True)
    return expr


def _query_name_tokens(query: str) -> list[str]:
    normalized = query.lower().replace("_", " ")
    for phrase in sorted(STAT_ALIASES, key=len, reverse=True):
        normalized = normalized.replace(phrase, " ")

    stat_words = set()
    for phrase in STAT_ALIASES:
        stat_words.update(phrase.split())

    words = []
    for word in re.findall(r"[a-zA-Z.'-]+", normalized):
        if len(word) <= 1 or word in QUERY_STOP_WORDS or word in stat_words:
            continue
        words.append(word)
    return words


def _player_name_column(df: pl.DataFrame) -> str | None:
    for column in ("player_display_name", "player_name", "display_name", "name"):
        if column in df.columns:
            return column
    return None


def _available_display_columns(df: pl.DataFrame, stat: str | None = None) -> list[str]:
    columns = [column for column in DISPLAY_COLUMNS if column in df.columns]
    if stat and stat in df.columns and stat not in columns:
        columns.append(stat)
    return columns


def find_player_rows(df: pl.DataFrame, query: str) -> pl.DataFrame:
    """Find rows whose player name appears in the query."""
    name_column = _player_name_column(df)
    if name_column is None:
        return df.head(0)

    words = _query_name_tokens(query)
    if not words:
        return df.head(0)

    matches = df.filter(_contains_words_expr(name_column, words))
    if len(matches) > 0:
        return matches

    names = [str(name) for name in df.select(name_column).unique().to_series().to_list()]
    scored_names = []
    for name in names:
        lowered = name.lower()
        score = sum(1 for word in words if word in lowered)
        if score > 0:
            scored_names.append((score, len(name), name))

    if not scored_names:
        return df.head(0)

    best_score = max(score for score, _, _ in scored_names)
    if best_score < min(2, len(words)):
        return df.head(0)

    best_names = [name for score, _, name in scored_names if score == best_score]
    return df.filter(pl.col(name_column).is_in(best_names))


def summarize_player(
    df: pl.DataFrame,
    query: str,
    stat: str | None = None,
    limit: int = 8,
) -> dict[str, Any] | None:
    """Summarize one matched player with recent games and simple projections."""
    matches = find_player_rows(df, query)
    if len(matches) == 0:
        return None

    name_column = _player_name_column(matches)
    if name_column is None:
        return None

    if stat is None:
        stat = best_player_stat(matches)

    names = [str(name) for name in matches.select(name_column).to_series().to_list()]
    player_name = Counter(names).most_common(1)[0][0]
    player_rows = matches.filter(pl.col(name_column) == player_name)
    sort_columns = [
        column for column in ("season", "week") if column in player_rows.columns
    ]
    if sort_columns:
        player_rows = player_rows.sort(sort_columns, descending=True)

    selected_columns = [name_column, *_available_display_columns(player_rows, stat)]
    selected_columns = list(dict.fromkeys(selected_columns))
    recent_games = player_rows.select(selected_columns).head(limit).to_dicts()
    projection = project_player_stat(player_rows, stat) if stat else None

    return {
        "type": "player",
        "title": player_name,
        "stat": stat,
        "projection": projection,
        "rows": recent_games,
        "summary": _player_summary_text(player_name, projection),
    }


def summarize_team(
    df: pl.DataFrame,
    query: str,
    stat: str | None = None,
    limit: int = 8,
) -> dict[str, Any] | None:
    team = _find_team_alias(query)
    if not team:
        return None

    team_col = None
    for candidate in ("recent_team", "team"):
        if candidate in df.columns:
            team_col = candidate
            break

    if team_col is None:
        return None

    if stat is None:
        stat = default_stat(df.columns)
    if not stat or stat not in df.columns:
        return None

    team_rows = df.filter(pl.col(team_col) == team)
    if len(team_rows) == 0:
        return None

    rows = []
    if "is_home" in df.columns:
        location = _query_team_location(query)
        if location == "home":
            team_rows = team_rows.filter(pl.col("is_home") == True)
        elif location == "away":
            team_rows = team_rows.filter(pl.col("is_home") == False)

        summary_df = team_rows.group_by("is_home").agg(
            [
                pl.sum(stat).alias(stat),
                pl.len().alias("games"),
                pl.mean(stat).alias(f"{stat}_per_game"),
            ]
        )
        for row in summary_df.sort("is_home", descending=True).to_dicts():
            location_label = "Home" if row.get("is_home") else "Away"
            rows.append(
                {
                    "location": location_label,
                    "games": int(row.get("games") or 0),
                    stat: round(float(row.get(stat) or 0), 2),
                    f"{stat}_per_game": round(float(row.get(f"{stat}_per_game") or 0), 2),
                }
            )
    else:
        total = float(team_rows.select(pl.col(stat).sum()).to_series()[0] or 0)
        games = len(team_rows)
        rows = [
            {
                "team": team,
                "games": games,
                stat: round(total, 2),
                f"{stat}_per_game": round(total / games, 2) if games else 0,
            }
        ]

    return {
        "type": "team",
        "title": f"{team} team stats",
        "stat": stat,
        "projection": None,
        "rows": rows,
        "summary": f"{team} team totals for {stat.replace('_', ' ')} across the loaded seasons.",
    }


def leaderboard(
    df: pl.DataFrame,
    query: str,
    stat: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Build a stat leaderboard from season-to-date player stats."""
    if stat is None:
        stat = default_stat(df.columns)
    name_column = _player_name_column(df)

    if not stat or stat not in df.columns or name_column is None:
        return {
            "type": "empty",
            "title": "No matching stat found",
            "summary": "Try a query like 'receiving yards leaders' or 'Josh Allen passing yards'.",
            "rows": [],
            "stat": stat,
            "projection": None,
        }

    rows: dict[tuple[str, str | None], float] = {}
    for row in df.select(
        [
            column
            for column in (name_column, "recent_team", stat)
            if column in df.columns
        ]
    ).to_dicts():
        key = (str(row[name_column]), row.get("recent_team"))
        rows[key] = rows.get(key, 0.0) + float(row.get(stat) or 0)

    leader_rows = [
        {
            name_column: player_name,
            **({"recent_team": team} if team is not None else {}),
            stat: round(total, 2),
        }
        for (player_name, team), total in rows.items()
    ]
    leader_rows.sort(key=lambda row: float(row[stat]), reverse=True)

    return {
        "type": "leaderboard",
        "title": f"Top {limit} by {stat.replace('_', ' ')}",
        "stat": stat,
        "projection": None,
        "rows": leader_rows[:limit],
        "summary": f"Ranked by total {stat.replace('_', ' ')} across the loaded seasons.",
    }


def project_player_stat(df: pl.DataFrame, stat: str | None) -> dict[str, Any] | None:
    """Project the next game using recent average plus a simple trend signal."""
    if not stat or stat not in df.columns:
        return None

    sort_columns = [column for column in ("season", "week") if column in df.columns]
    recent = df.sort(sort_columns, descending=True) if sort_columns else df
    values = [
        float(value)
        for value in recent.select(stat).drop_nulls().head(6).to_series().to_list()
        if isinstance(value, int | float)
    ]
    if not values:
        return None

    recent_avg = sum(values[:3]) / min(3, len(values))
    longer_avg = sum(values) / len(values)
    projection = round((recent_avg * 0.65) + (longer_avg * 0.35), 1)
    trend = round(recent_avg - longer_avg, 1)

    if trend > 0:
        direction = "up"
    elif trend < 0:
        direction = "down"
    else:
        direction = "flat"

    return {
        "stat": stat,
        "projection": projection,
        "recent_average": round(recent_avg, 1),
        "sample_average": round(longer_avg, 1),
        "trend": trend,
        "direction": direction,
        "sample_size": len(values),
        "method": "65% last three games, 35% last six available games",
    }
def _apply_split_filters(df: pl.DataFrame, params: dict[str, list[str]]) -> pl.DataFrame:
    """Apply split filters from query params to the player stats dataframe.

    See the nested version in previous edits for behavior.
    """
    if df is None or len(df) == 0:
        return df

    def has(col: str) -> bool:
        return col in df.columns

    # Location filter
    location = params.get("location", [None])[0]
    if location and has("is_home"):
        if location == "home":
            df = df.filter(pl.col("is_home") == True)
        elif location == "away":
            df = df.filter(pl.col("is_home") == False)

    # Roof filter
    roof = params.get("roof", [None])[0]
    if roof and has("roof"):
        normalized = roof.lower()
        if normalized == "indoors":
            df = df.filter(pl.col("roof").is_in(["dome", "closed"]))
        elif normalized == "outdoors":
            df = df.filter(pl.col("roof").is_in(["outdoors", "open"]))
        else:
            df = df.filter(pl.col("roof") == roof)

    # Weather filter dropdown
    weather = params.get("weather", [None])[0]
    if weather and has("weather"):
        normalized = weather.lower()
        if normalized == "rain":
            df = df.filter(pl.col("weather").str.to_lowercase().str.contains("rain"))
        elif normalized == "snow":
            df = df.filter(pl.col("weather").str.to_lowercase().str.contains("snow"))
        elif normalized in ("below_0", "below 0"):
            weather_text = pl.col("weather").str.to_lowercase()
            df = df.filter(
                weather_text.str.contains(r"below 0")
                | weather_text.str.contains(r"below zero")
                | weather_text.str.contains(r"-\d+")
                | weather_text.str.contains(r"\b0\s*°\b")
            )
        else:
            df = df.filter(pl.col("weather").str.to_lowercase().str.contains(normalized))

    # Day filter
    day = params.get("day", [None])[0]
    if day and has("game_date"):
      try:
        dow = pl.col("game_date").str.strptime(pl.Date, fmt="%Y-%m-%d").dt.weekday()
        name = day.lower()
        if name in ("weekday", "weekdays"):
          df = df.filter(dow.is_in([0, 1, 2, 3, 4]))
        elif name in ("weekend", "weekends"):
          df = df.filter(dow.is_in([5, 6]))
        else:
          mapping = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
          }
          if name in mapping:
            df = df.filter(dow == mapping[name])
      except Exception:
        pass

    # Time of day
    tod = params.get("time_of_day", [None])[0]
    if tod and has("game_time"):
        try:
            hour = pl.col("game_time").str.split(":").list.get(0).cast(pl.Int64)
            if tod == "europe":
                df = df.filter(hour < 13)
            elif tod == "early":
                df = df.filter(hour.is_in([13, 14]))
            elif tod == "midday":
                df = df.filter(hour.is_in([15, 16, 17]))
            elif tod == "night":
                df = df.filter(hour >= 18)
        except Exception:
            pass

    # Divisional filter
    div_flag = params.get("divisional", [None])[0]
    if div_flag and has("div_game"):
        df = df.filter(pl.col("div_game") == 1)

    return df


def _player_summary_text(player_name: str, projection: dict[str, Any] | None) -> str:
    if not projection:
        return f"Found recent rows for {player_name}."

    stat = projection["stat"].replace("_", " ")
    return (
        f"{player_name} projects around {projection['projection']} {stat} next game "
        f"from a {projection['sample_size']}-game sample; recent trend is "
        f"{projection['direction']}."
    )


def compare_players(
    df: pl.DataFrame,
    player_names: tuple[str, ...],
    stat: str | None = None,
) -> dict[str, Any] | None:
    if not player_names or len(player_names) < 2:
        return None

    if stat is None:
        stat = default_stat(df.columns)

    name_column = _player_name_column(df)
    if not stat or stat not in df.columns or name_column is None:
        return None

    results: list[dict[str, Any]] = []
    for player_name in player_names:
        player_rows = find_player_rows(df, player_name)
        if len(player_rows) == 0:
            continue

        canonical_name = Counter(
            str(name) for name in player_rows.select(name_column).to_series().to_list()
        ).most_common(1)[0][0]
        player_rows = player_rows.filter(pl.col(name_column) == canonical_name)
        total = float(player_rows.select(pl.col(stat).sum()).to_series()[0] or 0)
        games = len(player_rows)
        per_game = round(total / games, 2) if games else 0.0
        team = None
        if "recent_team" in player_rows.columns:
            teams = [str(team) for team in player_rows.select("recent_team").unique().to_series().to_list() if team]
            team = teams[0] if len(teams) == 1 else None

        results.append(
            {
                name_column: canonical_name,
                **({"recent_team": team} if team else {}),
                "games": games,
                stat: round(total, 2),
                f"{stat}_per_game": per_game,
            }
        )

    if len(results) < 2:
        return None

    results.sort(key=lambda row: float(row.get(stat, 0) or 0), reverse=True)
    return {
        "type": "compare",
        "title": f"Compare {len(results)} players by {stat.replace('_', ' ')}",
        "summary": f"Comparing {', '.join(str(row[name_column]) for row in results)} on {stat.replace('_', ' ')}.",
        "rows": results,
        "stat": stat,
        "projection": None,
    }


def answer_query(df: pl.DataFrame, config: SearchConfig) -> dict[str, Any]:
    """Return the best answer for a user's player/stat query."""
    query = config.query.strip()
    if not query and not config.compare:
        return {
            "type": "empty",
            "title": "Ask about a player or team",
            "summary": "Examples: 'Tyreek Hill receiving yards', 'Lions away rushing yards', or 'QB fantasy points PPR'.",
            "rows": [],
            "stat": None,
            "projection": None,
        }

    stat = infer_stat(query, df.columns)

    if config.compare:
        compare_result = compare_players(df, config.compare, stat=stat)
        if compare_result:
            return compare_result

    if query:
        team_result = summarize_team(df, query, stat=stat, limit=config.limit)
        if team_result:
            return team_result
        player_result = summarize_player(df, query, stat=stat, limit=config.limit)
        if player_result:
            return player_result

    return leaderboard(df, query, stat=stat, limit=config.limit)


@lru_cache(maxsize=8)
def load_recent_player_stats(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Load and cache weekly player stats for the web app."""
    df = load_player_stats(list(seasons), summary_level="week")

    # Attempt to enrich player rows with schedule-level fields for split filters.
    try:
      schedules = load_schedules(list(seasons))
    except Exception:
      return df

    # Build per-team schedule rows (one per team per game) for joining.
    def _select_columns(sched: pl.DataFrame, cols: list[str]):
        return [c for c in cols if c in sched.columns]

    sched_cols = [
        "season",
        "week",
        "home_team",
        "away_team",
        "div_game",
        "roof",
        "game_date",
        "game_time",
        "weather",
    ]

    # create home and away team views
    home_cols = _select_columns(schedules, sched_cols)
    try:
        home = schedules.select(home_cols).with_columns(
            pl.col("home_team").alias("team"),
            pl.col("away_team").alias("opponent_team"),
            pl.lit(True).alias("is_home"),
        )
        away = schedules.select(home_cols).with_columns(
            pl.col("away_team").alias("team"),
            pl.col("home_team").alias("opponent_team"),
            pl.lit(False).alias("is_home"),
        )
        sched_team = pl.concat([home, away], how="vertical")

        # join on season, week, and team/recent_team if available
        join_left = None
        if "recent_team" in df.columns:
            join_left = "recent_team"
        elif "team" in df.columns:
            join_left = "team"

        if join_left is not None:
            # perform left join; use suffix to avoid name collisions
            df = df.join(
                sched_team,
                left_on=["season", "week", join_left],
                right_on=["season", "week", "team"],
                how="left",
                suffix="_sched",
            )
    except Exception:
        # If any schedule joining step fails, return original df silently.
        return df

    return df


class NflReadPyUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the local UI."""

    server_version = "nflreadpy-ui/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/sleeper":
            params = parse_qs(parsed.query)
            username = params.get("username", [None])[0]
            season_value = params.get("season", [None])[0]
            league_id = params.get("league_id", [None])[0]
            try:
              if league_id:
                # Fetch league rosters and players and try to map to nflreadpy stats
                season = parse_seasons(season_value)[-1]
                rosters_resp = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/rosters")
                users_resp = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/users")
                players_resp = requests.get("https://api.sleeper.app/v1/players/nfl")
                if rosters_resp.status_code != 200:
                  raise ValueError(f"Sleeper rosters API error: {rosters_resp.status_code}")
                if users_resp.status_code != 200:
                  raise ValueError(f"Sleeper users API error: {users_resp.status_code}")
                if players_resp.status_code != 200:
                  raise ValueError(f"Sleeper players API error: {players_resp.status_code}")

                rosters = rosters_resp.json()
                users = users_resp.json()
                players_map = players_resp.json()

                # Build user lookup by user_id (string-safe)
                users_by_id = {str(u.get("user_id")): u for u in users}

                # Load a small recent stats dataframe to map player names -> rows
                try:
                  df = load_recent_player_stats(recent_seasons(1))
                except Exception:
                  df = None

                def find_player_stats(full_name: str):
                  if not full_name or df is None:
                    return []
                  # try exact match first
                  try:
                    mask = pl.col("player_display_name") == full_name
                    if df.filter(mask).is_empty() is False:
                      return df.filter(mask).select(["player_display_name", "season", "week", "receiving_yards", "rushing_yards", "fantasy_points"]).head(8).to_dicts()
                  except Exception:
                    pass
                  # fallback: match by last name
                  last = full_name.split()[-1]
                  try:
                    mask2 = pl.col("player_display_name").str.contains(last)
                    return df.filter(mask2).select(["player_display_name", "season", "week", "receiving_yards", "rushing_yards", "fantasy_points"]).head(6).to_dicts()
                  except Exception:
                    return []

                roster_list = []
                for r in rosters:
                  owner = users_by_id.get(str(r.get("owner_id")), {})
                  players = []
                  for pid in r.get("players", []):
                    info = players_map.get(str(pid)) or players_map.get(pid)
                    full_name = None
                    if isinstance(info, dict):
                      full_name = info.get("full_name") or info.get("fullName")
                    # best-effort stats mapping
                    stats = find_player_stats(full_name) if full_name else []
                    players.append({"player_id": pid, "full_name": full_name, "stats": stats})
                  roster_list.append({"roster_id": r.get("roster_id"), "owner_id": r.get("owner_id"), "owner_display_name": owner.get("display_name"), "players": players})

                self._send_json({"type": "sleeper_league", "league_id": league_id, "season": season, "rosters": roster_list})
              else:
                if not username:
                  raise ValueError("Missing username parameter")
                season = parse_seasons(season_value)[-1]
                # get user id
                user_resp = requests.get(f"https://api.sleeper.app/v1/user/{username}")
                if user_resp.status_code != 200:
                  raise ValueError(f"Sleeper API error: {user_resp.status_code}")
                user_id = user_resp.json().get("user_id")
                if not user_id:
                  raise ValueError("User not found")

                leagues_resp = requests.get(
                  f"https://api.sleeper.app/v1/user/{user_id}/leagues/nfl/{season}"
                )
                if leagues_resp.status_code != 200:
                  raise ValueError(f"Sleeper API error: {leagues_resp.status_code}")
                leagues = {l.get("name"): l.get("league_id") for l in leagues_resp.json()}
                self._send_json({"type": "sleeper", "username": username, "user_id": user_id, "leagues": leagues})
            except Exception as exc:  # pragma: no cover - visible in browser
              self._send_json({"type": "error", "title": "Could not fetch Sleeper data", "summary": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path == "/":
            self._send_html(render_page())
            return

        if parsed.path == "/api/suggest":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            season_value = params.get("seasons", [None])[0]
            try:
                seasons = parse_seasons(season_value)
                df = load_recent_player_stats(seasons)
                suggestions = suggest_names(df, query, limit=int(params.get("limit", ["20"])[0]))
                self._send_json({"type": "suggestions", "query": query, "suggestions": suggestions})
            except Exception as exc:  # pragma: no cover - visible in browser
                logging.error("Unhandled exception in /api/suggest: %s", exc)
                traceback.print_exc()
                self._send_json(
                    {
                        "type": "error",
                        "title": "Could not generate suggestions",
                        "summary": str(exc),
                        "suggestions": [],
                    },
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            season_value = params.get("seasons", [None])[0]
            try:
                seasons = parse_seasons(season_value)
                limit = int(params.get("limit", ["8"])[0])
                try:
                    df = load_recent_player_stats(seasons)
                except MemoryError:
                    logging.error("MemoryError while loading recent player stats for %s", seasons)
                    traceback.print_exc()
                    self._send_json(
                        {
                            "type": "error",
                            "title": "Server memory exhausted",
                            "summary": "Loading requested seasons used too much memory. Try a smaller seasons range or use /api/debug-sample.",
                            "rows": [],
                        },
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                    return

                # Apply optional split filters from query params
                df = _apply_split_filters(df, params)
                compare_names = parse_compare_names(params.get("compare", [""])[0])

                result = answer_query(
                    df,
                    SearchConfig(
                        query=query,
                        seasons=seasons,
                        limit=limit,
                        compare=compare_names,
                    ),
                )
                result["seasons"] = seasons
                self._send_json(result)
            except Exception as exc:  # pragma: no cover - visible in browser
                logging.error("Unhandled exception in /api/search: %s", exc)
                traceback.print_exc()
                self._send_json(
                    {
                        "type": "error",
                        "title": "Could not load NFL data",
                        "summary": str(exc),
                        "rows": [],
                    },
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/api/debug-sample":
          # Return a small sample of recent player stats to validate search flow
          try:
            seasons = recent_seasons(1)
            df = load_recent_player_stats(seasons)
            sample = df.head(30).to_dicts() if df is not None else []
            self._send_json({"type": "sample", "seasons": seasons, "rows": sample})
          except Exception as exc:
            logging.error("Error in /api/debug-sample: %s", exc)
            traceback.print_exc()
            self._send_json({"type": "error", "title": "Debug sample failed", "summary": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
          return

        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the local server quiet unless the caller wraps it."""

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        encoded = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def render_page() -> str:
    """Render the single-page app shell."""
    default_seasons = ", ".join(str(season) for season in recent_seasons())
    title = html.escape("nflreadpy Stat Finder")
    s = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17212b;
      --muted: #5b6673;
      --line: #d8dee6;
      --field: #f6f8fb;
      --accent: #0b6bcb;
      --accent-ink: #ffffff;
      --surface: #ffffff;
      --band: #eef3f7;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--band);
      color: var(--ink);
    }}
    header {{
      background: #123047;
      color: white;
      padding: 24px clamp(16px, 5vw, 56px);
    }}
    header h1 {{
      margin: 0 0 6px;
      font-size: clamp(28px, 4vw, 42px);
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      max-width: 760px;
      color: #d8e6f3;
      line-height: 1.5;
    }}
    main {{
      width: min(1120px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    .search-panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }}
    .search-grid {{
      display: grid;
      grid-template-columns: 1fr minmax(160px, 220px) auto;
      gap: 12px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }}
    input {{
      width: 100%;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--field);
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
    }}
    .autocomplete-wrapper {{
      position: relative;
    }}
    .suggestions {{
      position: absolute;
      top: calc(100% + 4px);
      left: 0;
      right: 0;
      background: var(--surface);
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 24px rgba(17, 24, 39, 0.08);
      z-index: 20;
      max-height: 256px;
      overflow-y: auto;
    }}
    .suggestion-item {{
      padding: 10px 12px;
      cursor: pointer;
      font: inherit;
      color: var(--ink);
      background: #ffffff;
      border-bottom: 1px solid #edf1f5;
    }}
    .suggestion-item:nth-child(odd) {{
      background: #f6f8fb;
    }}
    .suggestion-item:last-child {{
      border-bottom: none;
    }}
    .suggestion-item:hover {{
      background: #e8eff6;
    }}
    .suggestion-item.active {{
      background: #d7e4f0;
    }}
    button {{
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: var(--accent-ink);
      padding: 0 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .examples {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .examples button {{
      min-height: 32px;
      background: #e8f1fb;
      color: #17466d;
      border: 1px solid #c8d8e8;
      font-size: 13px;
    }}
    .status {{
      min-height: 24px;
      margin: 18px 0 10px;
      color: var(--muted);
    }}
    .answer {{
      display: grid;
      gap: 16px;
    }}
    .result-header, .projection {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .result-header h2 {{
      margin: 0 0 6px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .result-header p, .projection p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .metric strong {{
      font-size: 20px;
    }}
    .table-wrap {
      overflow-x: auto;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .league-list {
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 10px;
    }
    .league-list li {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
    }
    .league-button {
      border: 1px solid var(--accent);
      background: var(--accent);
      color: var(--accent-ink);
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
    }
    .league-roster {
      margin-top: 18px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
    }
    .close-button {
      border: none;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 16px;
      padding: 0;
      margin-bottom: 8px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .close-button:hover {
      color: var(--text);
    }
    .player-stats {
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      white-space: nowrap;
      font-size: 14px;
    }}
    th {{
      background: #f0f4f8;
      color: #33495f;
      font-size: 12px;
      text-transform: uppercase;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    @media (max-width: 760px) {{
      .search-grid {{
        grid-template-columns: 1fr;
      }}
      button {{
        width: 100%;
      }}
      .metrics {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>@@TITLE@@</h1>
    <p>Search recent nflreadpy player stats by player name or category, then get a simple recent-form projection from the loaded data.</p>
  </header>
  <main>
    <section class="search-panel">
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:12px;">
        <label style="margin:0;">
          Sleeper username
          <input id="sleeper_user" placeholder="Enter Sleeper username">
        </label>
        <button type="button" id="findLeagues">Find My Leagues</button>
        <div id="sleeperStatus" style="color:var(--muted);margin-left:12px"></div>
      </div>
      <form id="searchForm" class="search-grid">
        <label>
          Player or team name
          <div class="autocomplete-wrapper">
            <input id="query" name="query" autocomplete="off" placeholder="Enter player or team name">
            <div id="playerSuggestions" class="suggestions" hidden></div>
          </div>
        </label>
        <label>
          Compare players
          <input id="compare" name="compare" autocomplete="off" placeholder="Mahomes, Allen, Jefferson (comma / and / vs)">
          <small style="color: var(--muted); font-size: 12px;">Enter 2-3 player names separated by commas, “and”, or “vs”.</small>
        </label>
        <label>
          Seasons
          <input id="seasons" name="seasons" value="@@DEFAULT_SEASONS@@">
        </label>
        <label>
          Location
          <select id="location" name="location">
            <option value="all">All</option>
            <option value="home">Home</option>
            <option value="away">Away</option>
          </select>
        </label>
        <label>
          Roof
          <select id="roof" name="roof">
            <option value="">All</option>
            <option value="indoors">Indoors</option>
            <option value="outdoors">Outdoors</option>
          </select>
        </label>
        <label>
          Day
          <select id="day" name="day">
            <option value="">All</option>
            <option value="weekday">Weekday</option>
            <option value="weekend">Weekend</option>
            <option value="monday">Monday</option>
            <option value="tuesday">Tuesday</option>
            <option value="wednesday">Wednesday</option>
            <option value="thursday">Thursday</option>
            <option value="friday">Friday</option>
            <option value="saturday">Saturday</option>
            <option value="sunday">Sunday</option>
          </select>
        </label>
        <label>
          Time of day
          <select id="time_of_day" name="time_of_day">
            <option value="">All</option>
            <option value="europe">Europe</option>
            <option value="early">Early</option>
            <option value="midday">Midday</option>
            <option value="night">Night</option>
          </select>
        </label>
        <label>
          Weather
          <select id="weather" name="weather">
            <option value="">All</option>
            <option value="rain">Rain</option>
            <option value="snow">Snow</option>
            <option value="below_0">Below 0°</option>
          </select>
        </label>
        <label style="align-items:center;grid-auto-flow:column;gap:8px;">
          <input type="checkbox" id="divisional" name="divisional" value="1">
          Divisional rivals only
        </label>
        <button type="submit">Search</button>
      </form>
      <div class="examples">
        <button type="button" data-query="Patrick Mahomes passing yards">Mahomes passing</button>
        <button type="button" data-query="rushing yards leaders">Rushing leaders</button>
        <button type="button" data-query="Ja'Marr Chase fantasy points PPR">Chase fantasy</button>
      </div>
    </section>
    <div id="status" class="status"></div>
    <div id="sleeperResult" style="max-width:min(1120px,calc(100vw - 32px));margin:12px auto 0"></div>
    <section id="answer" class="answer"></section>
  </main>
  <script>
    const form = document.querySelector("#searchForm");
    const queryInput = document.querySelector("#query");
    const compareInput = document.querySelector("#compare");
    const playerList = document.querySelector("#playerSuggestions");
    const seasonsInput = document.querySelector("#seasons");
    const statusEl = document.querySelector("#status");
    const answerEl = document.querySelector("#answer");

    async function updateAutocomplete() {
      const query = queryInput.value.trim();
      if (!query) {
        playerList.innerHTML = "";
        return;
      }

      try {
        const resp = await fetch(`/api/suggest?q=${encodeURIComponent(query)}&seasons=${encodeURIComponent(seasonsInput.value)}&limit=20`);
        const payload = await resp.json();
        if (!resp.ok || !payload.suggestions) {
          playerList.hidden = true;
          return;
        }

        playerList.innerHTML = payload.suggestions
          .slice(0, 20)
          .map((value) => `
            <div class="suggestion-item" role="option" data-value="${escapeHtml(value)}">
              ${escapeHtml(value)}
            </div>
          `)
          .join("");
        playerList.hidden = payload.suggestions.length === 0;
      } catch (err) {
        playerList.hidden = true;
        // ignore autocomplete failures
      }
    }

    queryInput.addEventListener("input", debounce(updateAutocomplete, 250));
    queryInput.addEventListener("keydown", handleSuggestionKeyboard);
    document.addEventListener("click", (event) => {
      if (!event.target.closest(".autocomplete-wrapper")) {
        playerList.hidden = true;
      }
    });

    function debounce(fn, wait) {
      let timeout;
      return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), wait);
      };
    }

    function handleSuggestionClick(value) {
      queryInput.value = value;
      playerList.hidden = true;
      queryInput.focus();
    }

    playerList.addEventListener("click", (event) => {
      const item = event.target.closest(".suggestion-item");
      if (item) {
        handleSuggestionClick(item.dataset.value);
      }
    });

    function handleSuggestionKeyboard(event) {
      const items = Array.from(playerList.querySelectorAll(".suggestion-item"));
      if (!items.length || playerList.hidden) return;

      const active = playerList.querySelector(".suggestion-item.active");
      let index = active ? items.indexOf(active) : -1;

      if (event.key === "ArrowDown") {
        event.preventDefault();
        const next = items[Math.min(index + 1, items.length - 1)];
        setActiveSuggestion(next, items);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        const prev = items[Math.max(index - 1, 0)];
        setActiveSuggestion(prev, items);
      } else if (event.key === "Enter") {
        if (active) {
          event.preventDefault();
          handleSuggestionClick(active.dataset.value);
        }
      }
    }

    function setActiveSuggestion(item, items) {
      items.forEach((suggestion) => suggestion.classList.remove("active"));
      if (item) {
        item.classList.add("active");
        item.scrollIntoView({ block: "nearest" });
      }
    }

    document.querySelectorAll("[data-query]").forEach((button) => {
      button.addEventListener("click", () => {
        queryInput.value = button.dataset.query;
        form.requestSubmit();
      });
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      statusEl.textContent = "Loading nflreadpy data...";
      answerEl.innerHTML = "";
      const params = new URLSearchParams({
        q: queryInput.value,
        compare: compareInput?.value || '',
        seasons: seasonsInput.value,
        limit: "10",
        location: document.querySelector('#location')?.value || '',
        roof: document.querySelector('#roof')?.value || '',
        day: document.querySelector('#day')?.value || '',
        time_of_day: document.querySelector('#time_of_day')?.value || '',
        weather: document.querySelector('#weather')?.value || '',
        divisional: document.querySelector('#divisional')?.checked ? '1' : '',
      });
      try {{
        const response = await fetch(`/api/search?${{params.toString()}}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.summary || "Search failed");
        statusEl.textContent = `Loaded seasons: ${{(payload.seasons || []).join(", ")}}`;
        renderAnswer(payload);
      }} catch (error) {{
        statusEl.textContent = "";
        answerEl.innerHTML = `<div class="result-header"><h2>Could not answer that yet</h2><p>${{escapeHtml(error.message)}}</p></div>`;
      }}
    }});

    document.querySelector('#findLeagues')?.addEventListener('click', async () => {
      const user = document.querySelector('#sleeper_user')?.value || '';
      const status = document.querySelector('#sleeperStatus');
      const resultEl = document.querySelector('#sleeperResult');
      if (!user) {
        status.textContent = 'Enter a Sleeper username';
        return;
      }
      status.textContent = 'Fetching leagues...';
      resultEl.innerHTML = '';
      try {
        const resp = await fetch(`/api/sleeper?username=${{encodeURIComponent(user)}}`);
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.summary || 'Failed to fetch');
        status.textContent = '';
        const leagues = payload.leagues || {};
        if (!Object.keys(leagues).length) {
          resultEl.innerHTML = `<div class="result-header"><h2>No leagues found for ${escapeHtml(payload.username)}</h2></div>`;
          return;
        }
        resultEl.innerHTML = renderLeagueList(payload.username, leagues);
      } catch (err) {
        status.textContent = '';
        resultEl.innerHTML = `<div class="result-header"><h2>Error</h2><p>${escapeHtml(err.message)}</p></div>`;
      }
    });

    async function fetchLeagueDetails(leagueId) {
      const status = document.querySelector('#sleeperStatus');
      const resultEl = document.querySelector('#sleeperResult');
      const season = document.querySelector('#seasons')?.value || '';
      status.textContent = 'Fetching league roster details...';
      try {
        const resp = await fetch(`/api/sleeper?league_id=${{encodeURIComponent(leagueId)}}&season=${{encodeURIComponent(season)}}`);
        const payload = await resp.json();
        if (!resp.ok) throw new Error(payload.summary || 'Failed to fetch league details');
        status.textContent = '';
        resultEl.innerHTML = renderLeagueDetails(payload);
      } catch (err) {
        status.textContent = '';
        resultEl.innerHTML = `<div class="result-header"><h2>Error</h2><p>${escapeHtml(err.message)}</p></div>`;
      }
    }

    function renderLeagueList(username, leagues) {
      const items = Object.entries(leagues).map(([name, id]) => {
        return `<li><button type="button" class="league-button" data-league-id="${escapeHtml(id)}">${escapeHtml(name)}</button> <span>${escapeHtml(id)}</span></li>`;
      }).join('');
      return `<div class="result-header"><button type="button" class="close-button" onclick="closeSleeperSection()">× Close</button><h2>Leagues for ${escapeHtml(username)}</h2></div><ul class="league-list">${items}</ul><div id="leagueDetail"></div>`;
    }

    document.addEventListener('click', (event) => {
      const target = event.target;
      if (target.matches('.league-button')) {
        fetchLeagueDetails(target.dataset.leagueId);
      }
    });

    function renderLeagueDetails(payload) {
      if (!payload || payload.type !== 'sleeper_league') {
        return `<div class="result-header"><h2>League data unavailable</h2></div>`;
      }
      const rows = payload.rosters.map((roster) => {
        const playerRows = roster.players.map((player) => {
          const stats = player.stats || [];
          const statSummary = stats.length ? stats.map((row) => `${escapeHtml(row.player_display_name)} (${escapeHtml(row.season)} W${escapeHtml(row.week)}): ${escapeHtml(row.fantasy_points)} FP`).join('<br>') : 'No recent stats found';
          return `<li><strong>${escapeHtml(player.full_name || player.player_id)}</strong><div class="player-stats">${statSummary}</div></li>`;
        }).join('');
        return `<section class="league-roster"><h3>${escapeHtml(roster.owner_display_name || roster.owner_id)} (${escapeHtml(roster.roster_id)})</h3><ul>${playerRows}</ul></section>`;
      }).join('');
      return `<div class="result-header"><button type="button" class="close-button" onclick="closeSleeperSection()">× Close</button><h2>League ${escapeHtml(payload.league_id)}</h2><p>Season: ${escapeHtml(payload.season)}</p>${rows}</div>`;
    }

    function closeSleeperSection() {
      const resultEl = document.querySelector('#sleeperResult');
      const statusEl = document.querySelector('#sleeperStatus');
      if (resultEl) resultEl.innerHTML = '';
      if (statusEl) statusEl.textContent = '';
    }

    function renderAnswer(payload) {
      const projection = payload.projection ? `
        <div class="projection">
          <p>${escapeHtml(payload.projection.method)}</p>
          <div class="metrics">
            <div class="metric"><span>Projection</span><strong>${{payload.projection.projection}}</strong></div>
            <div class="metric"><span>Recent avg</span><strong>${{payload.projection.recent_average}}</strong></div>
            <div class="metric"><span>Sample avg</span><strong>${{payload.projection.sample_average}}</strong></div>
            <div class="metric"><span>Trend</span><strong>${{escapeHtml(payload.projection.direction)}}</strong></div>
          </div>
        </div>` : "";
      answerEl.innerHTML = `
        <div class="result-header">
          <h2>${{escapeHtml(payload.title)}}</h2>
          <p>${{escapeHtml(payload.summary || "")}}</p>
        </div>
        ${{projection}}
        ${{renderTable(payload.rows || [])}}
      `;
    }}

    function renderTable(rows) {
      if (!rows.length) return "";
      const columns = Object.keys(rows[0]);
      const head = columns.map((column) => `<th>${{escapeHtml(column.replaceAll("_", " "))}}</th>`).join("");
      const body = rows.map((row) => {{
        return `<tr>${{columns.map((column) => `<td>${{escapeHtml(row[column])}}</td>`).join("")}}</tr>`;
      }}).join("");
      return `<div class="table-wrap"><table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table></div>`;
    }}

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    form.requestSubmit();

  </script>
</body>
</html>"""
    # replace placeholders and convert doubled braces used for literal braces in the f-string
    s = s.replace('@@TITLE@@', title).replace('@@DEFAULT_SEASONS@@', default_seasons)
    s = s.replace('{{', '{').replace('}}', '}')
    return s

def run(host: str = "0.0.0.0", port: int | None = None) -> None:
    if port is None:
        port = int(os.environ.get("PORT", "8080"))
    """Start the local nflreadpy UI server."""
    server = ThreadingHTTPServer((host, port), NflReadPyUIHandler)
    url = f"http://{host}:{port}"
    print(f"nflreadpy UI running at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down nflreadpy UI")
    finally:
        server.server_close()


def main() -> None:
    """CLI entrypoint for the local UI."""
    parser = argparse.ArgumentParser(description="Run the nflreadpy local stats UI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
