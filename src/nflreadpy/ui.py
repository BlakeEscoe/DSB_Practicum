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

import polars as pl

from .load_stats import load_player_stats
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
    "fantasy points": "fantasy_points",
    "fantasy": "fantasy_points",
    "ppr": "fantasy_points_ppr",
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
    "fantasy_points",
    "fantasy_points_ppr",
]


@dataclass(frozen=True)
class SearchConfig:
    """User-facing query settings."""

    query: str
    seasons: tuple[int, ...]
    limit: int = 8


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


def _player_summary_text(player_name: str, projection: dict[str, Any] | None) -> str:
    if not projection:
        return f"Found recent rows for {player_name}."

    stat = projection["stat"].replace("_", " ")
    return (
        f"{player_name} projects around {projection['projection']} {stat} next game "
        f"from a {projection['sample_size']}-game sample; recent trend is "
        f"{projection['direction']}."
    )


def answer_query(df: pl.DataFrame, config: SearchConfig) -> dict[str, Any]:
    """Return the best answer for a user's player/stat query."""
    query = config.query.strip()
    if not query:
        return {
            "type": "empty",
            "title": "Ask about a player or stat",
            "summary": "Examples: 'Justin Jefferson receiving yards', 'QB fantasy points', or 'rushing yards leaders'.",
            "rows": [],
            "stat": None,
            "projection": None,
        }

    stat = infer_stat(query, df.columns)
    player_result = summarize_player(df, query, stat=stat, limit=config.limit)
    if player_result:
        return player_result
    return leaderboard(df, query, stat=stat, limit=config.limit)


@lru_cache(maxsize=8)
def load_recent_player_stats(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Load and cache weekly player stats for the web app."""
    return load_player_stats(list(seasons), summary_level="week")


class NflReadPyUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the local UI."""

    server_version = "nflreadpy-ui/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_page())
            return

        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            season_value = params.get("seasons", [None])[0]
            try:
                seasons = parse_seasons(season_value)
                limit = int(params.get("limit", ["8"])[0])
                df = load_recent_player_stats(seasons)
                result = answer_query(
                    df, SearchConfig(query=query, seasons=seasons, limit=limit)
                )
                result["seasons"] = seasons
                self._send_json(result)
            except Exception as exc:  # pragma: no cover - visible in browser
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
    return f"""<!doctype html>
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
    .table-wrap {{
      overflow-x: auto;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
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
    <h1>{title}</h1>
    <p>Search recent nflreadpy player stats by player name or category, then get a simple recent-form projection from the loaded data.</p>
  </header>
  <main>
    <section class="search-panel">
      <form id="searchForm" class="search-grid">
        <label>
          Player or stat
          <input id="query" name="query" autocomplete="off" value="Justin Jefferson receiving yards">
        </label>
        <label>
          Seasons
          <input id="seasons" name="seasons" value="{default_seasons}">
        </label>
        <button type="submit">Search</button>
      </form>
      <div class="examples">
        <button type="button" data-query="Patrick Mahomes passing yards">Mahomes passing</button>
        <button type="button" data-query="rushing yards leaders">Rushing leaders</button>
        <button type="button" data-query="Ja'Marr Chase fantasy points">Chase fantasy</button>
      </div>
    </section>
    <div id="status" class="status"></div>
    <section id="answer" class="answer"></section>
  </main>
  <script>
    const form = document.querySelector("#searchForm");
    const queryInput = document.querySelector("#query");
    const seasonsInput = document.querySelector("#seasons");
    const statusEl = document.querySelector("#status");
    const answerEl = document.querySelector("#answer");

    document.querySelectorAll("[data-query]").forEach((button) => {{
      button.addEventListener("click", () => {{
        queryInput.value = button.dataset.query;
        form.requestSubmit();
      }});
    }});

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      statusEl.textContent = "Loading nflreadpy data...";
      answerEl.innerHTML = "";
      const params = new URLSearchParams({{
        q: queryInput.value,
        seasons: seasonsInput.value,
        limit: "10",
      }});
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

    function renderAnswer(payload) {{
      const projection = payload.projection ? `
        <div class="projection">
          <p>${{escapeHtml(payload.projection.method)}}</p>
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

    function renderTable(rows) {{
      if (!rows.length) return "";
      const columns = Object.keys(rows[0]);
      const head = columns.map((column) => `<th>${{escapeHtml(column.replaceAll("_", " "))}}</th>`).join("");
      const body = rows.map((row) => {{
        return `<tr>${{columns.map((column) => `<td>${{escapeHtml(row[column])}}</td>`).join("")}}</tr>`;
      }}).join("");
      return `<div class="table-wrap"><table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table></div>`;
    }}

    function escapeHtml(value) {{
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
