import html
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st
from streamlit_searchbox import st_searchbox

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from nflreadpy.ui import (
        SearchConfig,
        answer_query,
        get_autocomplete_suggestions,
        load_recent_player_stats,
        parse_compare_names,
        parse_seasons,
        _apply_split_filters,
    )
    from nflreadpy.old_nfl_files.load_schedules import load_schedules
    from nflreadpy.old_nfl_files.load_teams import load_teams

    # load_snap_counts.py does `from .utils_date import ...`, but that module
    # actually lives one package up (nflreadpy.utils_date), so the import
    # raises on its own. Alias it in sys.modules before importing so the
    # relative import resolves, without touching that file.
    from nflreadpy import utils_date as _nflreadpy_utils_date

    sys.modules.setdefault("nflreadpy.old_nfl_files.utils_date", _nflreadpy_utils_date)
    from nflreadpy.old_nfl_files.load_snap_counts import load_snap_counts
except Exception as exc:
    st.error(f"Unable to import nflreadpy UI helpers: {exc}")
    st.stop()

from sleeper_connect import get_user_id, get_leagues, get_all_rosters

st.set_page_config(page_title="Fantasy Football Stat Explorer", layout="wide")

st.title("NFL Stat Finder")
st.markdown(
    "Search recent nflreadpy player stats with splits, comparisons, and Sleeper league integration."
)

# --- ESPN-style stat splits -------------------------------------------------
#
# The player-week stats dataset only carries raw counting stats (completions,
# attempts, carries, etc). These helpers derive the ESPN-style splits table:
# grouping a matched player's games into situational buckets (home/away,
# outcome, month, surface, ...) and rolling up the position-appropriate stat
# columns (passing/rushing/receiving) for each bucket. Sections that need data
# we don't load (play-by-play for "last 2 min"/longest play, depth charts for
# starter status) are intentionally omitted.


def _passer_rating(cmp_: float, att: float, yds: float, td: float, interceptions: float) -> float:
    if not att:
        return 0.0
    a = max(0.0, min(((cmp_ / att) - 0.3) * 5, 2.375))
    b = max(0.0, min(((yds / att) - 3) * 0.25, 2.375))
    c = max(0.0, min((td / att) * 20, 2.375))
    d = max(0.0, min(2.375 - ((interceptions / att) * 25), 2.375))
    return round(((a + b + c + d) / 6) * 100, 1)


def _passing_group(sums: dict[str, float], games: int) -> dict[str, Any]:
    cmp_ = sums.get("completions", 0.0)
    att = sums.get("attempts", 0.0)
    yds = sums.get("passing_yards", 0.0)
    td = sums.get("passing_tds", 0.0)
    interceptions = sums.get("passing_interceptions", 0.0)
    sacks = sums.get("sacks_suffered", 0.0)
    g = games or 1
    # Counting stats are shown as per-game averages; CMP%/AVG/RTG are ratios,
    # so they come out identical whether computed from totals or per-game
    # rates and are left as season-wide rates.
    return {
        "CMP": round(cmp_ / g, 1),
        "ATT": round(att / g, 1),
        "YDS": round(yds / g, 1),
        "CMP%": round(cmp_ / att * 100, 1) if att else 0.0,
        "AVG": round(yds / att, 1) if att else 0.0,
        "TD": round(td / g, 1),
        "INT": round(interceptions / g, 1),
        "SACK": round(sacks / g, 1),
        "RTG": _passer_rating(cmp_, att, yds, td, interceptions),
    }


def _rushing_group(sums: dict[str, float], games: int) -> dict[str, Any]:
    car = sums.get("carries", 0.0)
    yds = sums.get("rushing_yards", 0.0)
    td = sums.get("rushing_tds", 0.0)
    g = games or 1
    return {
        "CAR": round(car / g, 1),
        "YDS": round(yds / g, 1),
        "AVG": round(yds / car, 1) if car else 0.0,
        "TD": round(td / g, 1),
    }


def _receiving_group(sums: dict[str, float], games: int) -> dict[str, Any]:
    rec = sums.get("receptions", 0.0)
    tgt = sums.get("targets", 0.0)
    yds = sums.get("receiving_yards", 0.0)
    td = sums.get("receiving_tds", 0.0)
    g = games or 1
    return {
        "REC": round(rec / g, 1),
        "TGT": round(tgt / g, 1),
        "YDS": round(yds / g, 1),
        "AVG": round(yds / rec, 1) if rec else 0.0,
        "TD": round(td / g, 1),
        "CATCH%": round(rec / tgt * 100, 1) if tgt else 0.0,
    }


STAT_GROUP_BUILDERS: dict[str, tuple[Any, list[str]]] = {
    "passing": (
        _passing_group,
        ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions", "sacks_suffered"],
    ),
    "rushing": (_rushing_group, ["carries", "rushing_yards", "rushing_tds"]),
    "receiving": (_receiving_group, ["receptions", "targets", "receiving_yards", "receiving_tds"]),
}

STAT_GROUP_HEADERS: dict[str, list[str]] = {
    "passing": ["CMP", "ATT", "YDS", "CMP%", "AVG", "TD", "INT", "SACK", "RTG"],
    "rushing": ["CAR", "YDS", "AVG", "TD"],
    "receiving": ["REC", "TGT", "YDS", "AVG", "TD", "CATCH%"],
}

STAT_GROUP_LABELS: dict[str, str] = {
    "passing": "PASSING",
    "rushing": "RUSHING",
    "receiving": "RECEIVING",
}

# (section label, bucket column, canonical row order)
SPLIT_SECTION_DEFS: list[tuple[str, str, list[str]]] = [
    ("OUTCOME", "outcome", ["Wins/Ties", "Losses"]),
    ("VICTORY MARGIN", "victory_margin", ["0-7", "8-14", "15+"]),
    ("SEASON GAMES", "season_game_block", ["1-8", "9-16"]),
    ("MONTH", "month", ["September", "October", "November", "December", "January", "February"]),
    ("DAY", "day_bucket", ["Sunday", "Monday", "Other"]),
    ("SURFACE", "surface_bucket", ["Grass", "Turf"]),
    ("LOCATION", "roof_label", ["Outdoors", "Indoors"]),
    ("WEATHER", "temperature_category", ["<40 F", "40-80 F", "81+ F"]),
    ("GROUP", "group_label", ["vs AFC", "vs NFC", "vs Div"]),
]


OFFENSIVE_POSITIONS = {"QB", "RB", "FB", "HB", "WR", "TE"}


def _player_stat_groups(position: str | None, totals: dict[str, float]) -> list[str]:
    position = (position or "").upper()
    groups: list[str] = []
    if position == "QB":
        groups.append("passing")
        if totals.get("carries"):
            groups.append("rushing")
    elif position in ("RB", "FB", "HB"):
        groups.append("rushing")
        if totals.get("targets"):
            groups.append("receiving")
    elif position in ("WR", "TE"):
        groups.append("receiving")
        if totals.get("carries"):
            groups.append("rushing")

    if not groups:
        for group, (_, cols) in STAT_GROUP_BUILDERS.items():
            if any(totals.get(col) for col in cols):
                groups.append(group)

    return groups


def _aggregate_group_row(rows: pl.DataFrame, group: str) -> dict[str, Any]:
    builder, cols = STAT_GROUP_BUILDERS[group]
    available = [c for c in cols if c in rows.columns]
    sums = {c: 0.0 for c in cols}
    if available:
        agg = rows.select([pl.sum(c).alias(c) for c in available]).to_dicts()[0]
        sums.update({c: float(agg.get(c) or 0) for c in available})
    return builder(sums, len(rows))


_FPTS_COLUMNS = [
    "passing_yards", "passing_tds", "passing_interceptions", "passing_2pt_conversions",
    "rushing_yards", "rushing_tds", "rushing_2pt_conversions",
    "receptions", "receiving_yards", "receiving_tds", "receiving_2pt_conversions",
    "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost",
]


def _bucket_fpts_per_game(rows: pl.DataFrame) -> float:
    """Average Sleeper PPR fantasy points per game for a bucket of games.

    Summing raw stats first and scoring once is equivalent to averaging each
    game's FPTS individually, since the scoring formula is linear.
    """
    if len(rows) == 0:
        return 0.0
    available = [c for c in _FPTS_COLUMNS if c in rows.columns]
    if not available:
        return 0.0
    sums = rows.select([pl.sum(c).alias(c) for c in available]).to_dicts()[0]
    return round(_sleeper_ppr_points(sums) / len(rows), 2)


def _derive_split_buckets(player_rows: pl.DataFrame) -> pl.DataFrame:
    """Compute the situational bucket columns the splits sections group by.

    Self-contained rather than reusing ui.py's _add_split_columns, which calls
    polars APIs (strptime(fmt=...), str.title()) that were removed in the
    polars version installed here and raise for any frame carrying a "roof"
    or "game_date" column.
    """
    if "game_date" in player_rows.columns:
        player_rows = player_rows.with_columns(
            pl.col("game_date")
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            .dt.strftime("%B")
            .alias("month")
        )

    if "weekday" in player_rows.columns:
        player_rows = player_rows.with_columns(
            pl.when(pl.col("weekday") == "Sunday")
            .then(pl.lit("Sunday"))
            .when(pl.col("weekday") == "Monday")
            .then(pl.lit("Monday"))
            .otherwise(pl.lit("Other"))
            .alias("day_bucket")
        )

    if "surface" in player_rows.columns:
        player_rows = player_rows.with_columns(
            pl.when(pl.col("surface").cast(pl.Utf8).str.to_lowercase() == "grass")
            .then(pl.lit("Grass"))
            .when(pl.col("surface").is_not_null())
            .then(pl.lit("Turf"))
            .otherwise(None)
            .alias("surface_bucket")
        )

    if "roof" in player_rows.columns:
        player_rows = player_rows.with_columns(
            pl.when(pl.col("roof").is_in(["dome", "closed"]))
            .then(pl.lit("Indoors"))
            .when(pl.col("roof").is_in(["outdoors", "open"]))
            .then(pl.lit("Outdoors"))
            .otherwise(None)
            .alias("roof_label")
        )

    if "temp" in player_rows.columns:
        player_rows = player_rows.with_columns(
            pl.when(pl.col("temp").is_null())
            .then(None)
            .when(pl.col("temp").cast(pl.Float64) < 40)
            .then(pl.lit("<40 F"))
            .when(pl.col("temp").cast(pl.Float64) >= 81)
            .then(pl.lit("81+ F"))
            .otherwise(pl.lit("40-80 F"))
            .alias("temperature_category")
        )

    if "is_home" in player_rows.columns:
        player_rows = player_rows.with_columns(
            pl.when(pl.col("is_home") == True)
            .then(pl.lit("Home"))
            .otherwise(pl.lit("Away"))
            .alias("location_label")
        )

    if "week" in player_rows.columns:
        player_rows = player_rows.with_columns(
            pl.when(pl.col("week") <= 8).then(pl.lit("1-8")).otherwise(pl.lit("9-16")).alias("season_game_block")
        )

    team_col = "recent_team" if "recent_team" in player_rows.columns else "team" if "team" in player_rows.columns else None
    has_scores = "home_score" in player_rows.columns and "away_score" in player_rows.columns
    has_teams = "home_team" in player_rows.columns and "away_team" in player_rows.columns
    if team_col and has_teams and has_scores:
        team_score = pl.when(pl.col(team_col) == pl.col("home_team")).then(pl.col("home_score")).otherwise(pl.col("away_score"))
        opp_score = pl.when(pl.col(team_col) == pl.col("home_team")).then(pl.col("away_score")).otherwise(pl.col("home_score"))
        score_diff = (team_score - opp_score).abs()
        player_rows = player_rows.with_columns(
            pl.when(team_score >= opp_score).then(pl.lit("Wins/Ties")).otherwise(pl.lit("Losses")).alias("outcome"),
            pl.when(score_diff <= 7)
            .then(pl.lit("0-7"))
            .when(score_diff <= 14)
            .then(pl.lit("8-14"))
            .otherwise(pl.lit("15+"))
            .alias("victory_margin"),
        )

        has_conf_div = all(
            c in player_rows.columns for c in ("home_conf", "away_conf", "home_division", "away_division")
        )
        if team_col and has_teams and has_conf_div:
            opponent_conf = pl.when(pl.col(team_col) == pl.col("home_team")).then(pl.col("away_conf")).otherwise(pl.col("home_conf"))
            team_division = pl.when(pl.col(team_col) == pl.col("home_team")).then(pl.col("home_division")).otherwise(pl.col("away_division"))
            opponent_division = pl.when(pl.col(team_col) == pl.col("home_team")).then(pl.col("away_division")).otherwise(pl.col("home_division"))
            player_rows = player_rows.with_columns(
                pl.when(team_division == opponent_division)
                .then(pl.lit("vs Div"))
                .when(opponent_conf == "AFC")
                .then(pl.lit("vs AFC"))
                .otherwise(pl.lit("vs NFC"))
                .alias("group_label")
            )

    return player_rows


def compute_player_splits(df: pl.DataFrame, player_name: str) -> dict[str, Any] | None:
    """Build an ESPN-style multi-section splits table for a single matched player."""
    name_column = next(
        (c for c in ("player_display_name", "player_name", "display_name", "name") if c in df.columns),
        None,
    )
    if name_column is None:
        return None

    player_rows = df.filter(pl.col(name_column) == player_name)
    if len(player_rows) == 0:
        return None

    player_rows = _derive_split_buckets(player_rows)

    position = None
    if "position" in player_rows.columns:
        positions = [str(p) for p in player_rows.select("position").to_series().to_list() if p]
        if positions:
            position = Counter(positions).most_common(1)[0][0]

    totals_cols = {c for _, cols in STAT_GROUP_BUILDERS.values() for c in cols}
    available_totals = [c for c in totals_cols if c in player_rows.columns]
    totals = (
        player_rows.select([pl.sum(c).alias(c) for c in available_totals]).to_dicts()[0]
        if available_totals
        else {}
    )

    groups = _player_stat_groups(position, totals)
    if not groups:
        return None

    def build_rows(bucket_col: str, order: list[str]) -> list[dict[str, Any]]:
        if bucket_col not in player_rows.columns:
            return []
        values = list(player_rows.select(bucket_col).drop_nulls().unique().to_series().to_list())
        ordered = [v for v in order if v in values] + sorted(v for v in values if v not in order)
        rows = []
        for label in ordered:
            subset = player_rows.filter(pl.col(bucket_col) == label)
            if len(subset) == 0:
                continue
            row: dict[str, Any] = {
                "label": label,
                "games": len(subset),
                "fpts_per_game": _bucket_fpts_per_game(subset),
            }
            for group in groups:
                row[group] = _aggregate_group_row(subset, group)
            rows.append(row)
        return rows

    all_row: dict[str, Any] = {
        "label": "All Splits",
        "games": len(player_rows),
        "fpts_per_game": _bucket_fpts_per_game(player_rows),
    }
    for group in groups:
        all_row[group] = _aggregate_group_row(player_rows, group)

    sections = [{"label": "SPLIT", "rows": [all_row] + build_rows("location_label", ["Home", "Away"])}]
    for label, column, order in SPLIT_SECTION_DEFS:
        rows = build_rows(column, order)
        if rows:
            sections.append({"label": label, "rows": rows})

    return {"player": player_name, "position": position, "groups": groups, "sections": sections}


def render_splits_table(splits: dict[str, Any]) -> str:
    groups = splits.get("groups", [])
    if not groups:
        return ""

    group_header_cells = []
    col_headers: list[str] = []
    for group in groups:
        headers = STAT_GROUP_HEADERS.get(group, [])
        group_header_cells.append(
            f'<th colspan="{len(headers)}">{html.escape(STAT_GROUP_LABELS.get(group, group.upper()))}</th>'
        )
        col_headers.extend(headers)

    thead = (
        "<thead>"
        f'<tr><th class="split-label-col" colspan="2"></th>{"".join(group_header_cells)}</tr>'
        f'<tr><th class="split-label-col">SPLIT</th><th>FPTS/G</th>'
        f'{"".join(f"<th>{html.escape(h)}</th>" for h in col_headers)}</tr>'
        "</thead>"
    )

    colspan = 2 + len(col_headers)
    body_rows = []
    for section in splits.get("sections", []):
        body_rows.append(
            f'<tr class="split-section-row"><th colspan="{colspan}">{html.escape(section["label"])}</th></tr>'
        )
        for row in section.get("rows", []):
            cells = [
                f'<td class="split-label-col">{html.escape(str(row["label"]))}</td>',
                f'<td class="split-fpts">{row.get("fpts_per_game", 0)}</td>',
            ]
            for group in groups:
                values = row.get(group, {})
                for header in STAT_GROUP_HEADERS.get(group, []):
                    cells.append(f"<td>{html.escape(str(values.get(header, '')))}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div class="espn-splits-wrapper">
      <style>
        .espn-splits-wrapper {{ overflow-x: auto; margin: 0.5rem 0 1rem; }}
        .espn-splits-table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; white-space: nowrap; }}
        .espn-splits-table th, .espn-splits-table td {{
          padding: 6px 10px; border-bottom: 1px solid rgba(128,128,128,0.25); text-align: right;
        }}
        .espn-splits-table th.split-label-col, .espn-splits-table td.split-label-col {{
          text-align: left; font-weight: 600;
        }}
        .espn-splits-table td.split-fpts {{ font-weight: 700; background: rgba(45,212,191,0.15); }}
        .espn-splits-table thead tr:first-child th {{
          text-align: center; border-bottom: 1px solid rgba(128,128,128,0.4);
        }}
        .espn-splits-table thead tr:last-child th {{
          font-weight: 700; font-size: 0.75rem; letter-spacing: 0.02em;
          border-bottom: 2px solid rgba(128,128,128,0.5);
        }}
        .espn-splits-table .split-section-row th {{
          text-align: left; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.04em;
          background: rgba(128,128,128,0.14); padding-top: 8px; padding-bottom: 8px;
        }}
        .espn-splits-table tbody tr:not(.split-section-row):nth-of-type(even) {{
          background: rgba(128,128,128,0.06);
        }}
      </style>
      <table class="espn-splits-table">
        {thead}
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


def _enrich_with_schedule_details(df: pl.DataFrame, seasons: tuple[int, ...]) -> pl.DataFrame:
    """Attach schedule fields the shared loader doesn't join (surface, temp,
    weekday, scores, conference/division) so the splits table has real data
    for SURFACE, WEATHER, OUTCOME, and GROUP.
    """
    if df is None or len(df) == 0 or "home_team" not in df.columns or "away_team" not in df.columns:
        return df

    try:
        schedules = load_schedules(list(seasons))
    except Exception:
        return df

    schedules = schedules.with_columns(
        pl.col("gameday").alias("game_date"),
        pl.col("gametime").alias("game_time"),
    )

    try:
        teams = load_teams().select(["team_abbr", "team_conf", "team_division"])
        schedules = schedules.join(
            teams.rename({"team_abbr": "home_team", "team_conf": "home_conf", "team_division": "home_division"}),
            on="home_team",
            how="left",
        ).join(
            teams.rename({"team_abbr": "away_team", "team_conf": "away_conf", "team_division": "away_division"}),
            on="away_team",
            how="left",
        )
    except Exception:
        pass

    extra_cols = [
        c
        for c in (
            "surface",
            "temp",
            "weekday",
            "game_date",
            "home_score",
            "away_score",
            "home_conf",
            "away_conf",
            "home_division",
            "away_division",
        )
        if c in schedules.columns
    ]
    if not extra_cols:
        return df

    try:
        sched_extra = schedules.select(["season", "week", "home_team", "away_team", *extra_cols]).unique(
            subset=["season", "week", "home_team", "away_team"]
        )
        return df.join(sched_extra, on=["season", "week", "home_team", "away_team"], how="left")
    except Exception:
        return df


# --- End ESPN-style stat splits ---------------------------------------------


# --- Sleeper standard PPR scoring + per-game log -----------------------------
#
# Sleeper's default "PPR" preset (confirmed via Sleeper support docs): 0.04
# pts/passing yard, 4 pts/passing TD, -1 pt/INT, 0.1 pts/rushing or receiving
# yard, 6 pts/rushing or receiving TD, 1 pt/reception, 2 pts/2-pt conversion,
# -2 pts/fumble lost. This mirrors nflverse's own fantasy_points_ppr formula
# except nflverse weights INT at -2; we compute it from raw counting stats so
# the FPTS numbers match Sleeper's ruleset exactly rather than nflverse's.
SLEEPER_PPR_SCORING = {
    "pass_yd": 0.04,
    "pass_td": 4,
    "pass_int": -1,
    "pass_2pt": 2,
    "rush_yd": 0.1,
    "rush_td": 6,
    "rush_2pt": 2,
    "rec": 1,
    "rec_yd": 0.1,
    "rec_td": 6,
    "rec_2pt": 2,
    "fum_lost": -2,
}

GAME_LOG_HEADERS: dict[str, list[str]] = {
    "passing": ["ATT", "CMP", "YD", "TD", "INT"],
    "rushing": ["ATT", "YD", "YPC", "TD"],
    "receiving": ["TAR", "REC", "YD", "YPT", "YPC", "TD"],
    "sacked": ["SK", "YDS"],
    "fumble": ["FUM", "LOST"],
    "returning": ["KR", "KYD", "PR", "PYD"],
}

GAME_LOG_LABELS: dict[str, str] = {
    "passing": "PASS",
    "rushing": "RUSHING",
    "receiving": "RECEIVING",
    "sacked": "SACKED",
    "fumble": "FUMBLE",
    "returning": "RETURNING",
}


def _sleeper_ppr_points(row: dict[str, Any]) -> float:
    def n(col: str) -> float:
        return float(row.get(col) or 0)

    points = (
        n("passing_yards") * SLEEPER_PPR_SCORING["pass_yd"]
        + n("passing_tds") * SLEEPER_PPR_SCORING["pass_td"]
        + n("passing_interceptions") * SLEEPER_PPR_SCORING["pass_int"]
        + n("passing_2pt_conversions") * SLEEPER_PPR_SCORING["pass_2pt"]
        + n("rushing_yards") * SLEEPER_PPR_SCORING["rush_yd"]
        + n("rushing_tds") * SLEEPER_PPR_SCORING["rush_td"]
        + n("rushing_2pt_conversions") * SLEEPER_PPR_SCORING["rush_2pt"]
        + n("receptions") * SLEEPER_PPR_SCORING["rec"]
        + n("receiving_yards") * SLEEPER_PPR_SCORING["rec_yd"]
        + n("receiving_tds") * SLEEPER_PPR_SCORING["rec_td"]
        + n("receiving_2pt_conversions") * SLEEPER_PPR_SCORING["rec_2pt"]
    )
    fumbles_lost = n("rushing_fumbles_lost") + n("receiving_fumbles_lost") + n("sack_fumbles_lost")
    points += fumbles_lost * SLEEPER_PPR_SCORING["fum_lost"]
    return round(points, 2)


def _game_log_group_values(row: dict[str, Any], group: str) -> dict[str, Any]:
    def n(col: str) -> float:
        return float(row.get(col) or 0)

    if group == "passing":
        return {
            "ATT": int(n("attempts")),
            "CMP": int(n("completions")),
            "YD": int(n("passing_yards")),
            "TD": int(n("passing_tds")),
            "INT": int(n("passing_interceptions")),
        }
    if group == "rushing":
        car, yds = n("carries"), n("rushing_yards")
        return {
            "ATT": int(car),
            "YD": int(yds),
            "YPC": round(yds / car, 1) if car else 0.0,
            "TD": int(n("rushing_tds")),
        }
    if group == "receiving":
        tar, rec, yds = n("targets"), n("receptions"), n("receiving_yards")
        return {
            "TAR": int(tar),
            "REC": int(rec),
            "YD": int(yds),
            "YPT": round(yds / tar, 2) if tar else 0.0,
            "YPC": round(yds / rec, 2) if rec else 0.0,
            "TD": int(n("receiving_tds")),
        }
    if group == "sacked":
        return {"SK": int(n("sacks_suffered")), "YDS": int(n("sack_yards_lost"))}
    if group == "fumble":
        fum = n("rushing_fumbles") + n("receiving_fumbles") + n("sack_fumbles")
        lost = n("rushing_fumbles_lost") + n("receiving_fumbles_lost") + n("sack_fumbles_lost")
        return {"FUM": int(fum), "LOST": int(lost)}
    if group == "returning":
        return {
            "KR": int(n("kickoff_returns")),
            "KYD": int(n("kickoff_return_yards")),
            "PR": int(n("punt_returns")),
            "PYD": int(n("punt_return_yards")),
        }
    return {}


def _attach_snap_pct(player_rows: pl.DataFrame, seasons: tuple[int, ...]) -> pl.DataFrame:
    """Best-effort join of offensive snap % from PFR snap counts by
    season/week/team/player name. Silently no-ops if anything doesn't line up
    (no shared player id exists between the two datasets).
    """
    name_col = next(
        (c for c in ("player_display_name", "player_name") if c in player_rows.columns), None
    )
    team_col = "recent_team" if "recent_team" in player_rows.columns else "team" if "team" in player_rows.columns else None
    if name_col is None or team_col is None:
        return player_rows

    try:
        snaps = load_snap_counts(list(seasons))
    except Exception:
        return player_rows

    if "player" not in snaps.columns or "offense_pct" not in snaps.columns:
        return player_rows

    snaps_small = snaps.select(["season", "week", "team", "player", "offense_pct"]).rename(
        {"team": "_snap_team", "player": "_snap_player", "offense_pct": "snap_pct"}
    )
    try:
        return player_rows.join(
            snaps_small,
            left_on=["season", "week", team_col, name_col],
            right_on=["season", "week", "_snap_team", "_snap_player"],
            how="left",
        )
    except Exception:
        return player_rows


def compute_game_log(df: pl.DataFrame, player_name: str, seasons: tuple[int, ...]) -> dict[str, Any] | None:
    """Build a per-game log (most recent first) with Sleeper standard PPR
    fantasy points and position-appropriate box-score columns.
    """
    name_column = next(
        (c for c in ("player_display_name", "player_name", "display_name", "name") if c in df.columns),
        None,
    )
    if name_column is None:
        return None

    player_rows = df.filter(pl.col(name_column) == player_name)
    if len(player_rows) == 0:
        return None

    player_rows = _attach_snap_pct(player_rows, seasons)

    sort_cols = [c for c in ("season", "week") if c in player_rows.columns]
    if sort_cols:
        player_rows = player_rows.sort(sort_cols, descending=True)

    position = None
    if "position" in player_rows.columns:
        positions = [str(p) for p in player_rows.select("position").to_series().to_list() if p]
        if positions:
            position = Counter(positions).most_common(1)[0][0]

    row_dicts = player_rows.to_dicts()
    totals = {
        "carries": sum(float(r.get("carries") or 0) for r in row_dicts),
        "targets": sum(float(r.get("targets") or 0) for r in row_dicts),
    }
    groups = _player_stat_groups(position, totals)
    if not groups:
        return None

    if "passing" in groups:
        groups = [*groups, "sacked"]
    has_fumbles = any(
        (r.get("rushing_fumbles") or r.get("receiving_fumbles") or r.get("sack_fumbles")) for r in row_dicts
    )
    if has_fumbles:
        groups = [*groups, "fumble"]
    has_returns = any((r.get("kickoff_returns") or r.get("punt_returns")) for r in row_dicts)
    if has_returns:
        groups = [*groups, "returning"]

    multi_season = len({r.get("season") for r in row_dicts if r.get("season") is not None}) > 1

    rows = []
    for row in row_dicts:
        opponent = row.get("opponent_team") or ""
        opponent_label = opponent if row.get("is_home") else f"@{opponent}" if opponent else ""
        snap_pct = row.get("snap_pct")
        entry: dict[str, Any] = {
            "season": row.get("season"),
            "week": row.get("week"),
            "opponent": opponent_label,
            "fpts": _sleeper_ppr_points(row),
            "snap_pct": round(float(snap_pct) * 100, 0) if snap_pct is not None else None,
        }
        for group in groups:
            entry[group] = _game_log_group_values(row, group)
        rows.append(entry)

    return {
        "player": player_name,
        "position": position,
        "groups": groups,
        "multi_season": multi_season,
        "rows": rows,
    }


def _format_stat_cell(value: Any) -> str:
    """Render zero-valued box-score stats as "-", matching typical game-log
    display convention (a made-up "0" reads as noise next to real zeros)."""
    if value is None:
        return "-"
    if isinstance(value, (int, float)) and value == 0:
        return "-"
    return str(value)


def render_game_log_table(game_log: dict[str, Any]) -> str:
    groups = game_log.get("groups", [])
    rows = game_log.get("rows", [])
    if not groups or not rows:
        return ""

    lead_cols = (["SEASON"] if game_log.get("multi_season") else []) + ["WK", "OPP", "FPTS", "SNP%"]

    group_header_cells = [f'<th colspan="{len(lead_cols)}"></th>']
    col_headers: list[str] = []
    for group in groups:
        headers = GAME_LOG_HEADERS.get(group, [])
        group_header_cells.append(
            f'<th colspan="{len(headers)}">{html.escape(GAME_LOG_LABELS.get(group, group.upper()))}</th>'
        )
        col_headers.extend(headers)

    thead = (
        "<thead>"
        f'<tr>{"".join(group_header_cells)}</tr>'
        f'<tr>{"".join(f"<th>{html.escape(h)}</th>" for h in lead_cols)}'
        f'{"".join(f"<th>{html.escape(h)}</th>" for h in col_headers)}</tr>'
        "</thead>"
    )

    body_rows = []
    for row in rows:
        cells = []
        if game_log.get("multi_season"):
            cells.append(f'<td class="game-log-label">{html.escape(str(row.get("season", "")))}</td>')
        cells.append(f'<td class="game-log-label">{html.escape(str(row.get("week", "")))}</td>')
        cells.append(f'<td class="game-log-label">{html.escape(str(row.get("opponent", "")))}</td>')
        cells.append(f'<td class="game-log-fpts">{row.get("fpts", 0)}</td>')
        snap_pct = row.get("snap_pct")
        cells.append(f'<td>{f"{int(snap_pct)}%" if snap_pct is not None else "-"}</td>')
        for group in groups:
            values = row.get(group, {})
            for header in GAME_LOG_HEADERS.get(group, []):
                cells.append(f"<td>{html.escape(_format_stat_cell(values.get(header)))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div class="game-log-wrapper">
      <style>
        .game-log-wrapper {{ overflow-x: auto; margin: 0.5rem 0 1rem; }}
        .game-log-table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; white-space: nowrap; }}
        .game-log-table th, .game-log-table td {{
          padding: 6px 10px; border-bottom: 1px solid rgba(128,128,128,0.25); text-align: right;
        }}
        .game-log-table td.game-log-label {{ text-align: left; font-weight: 600; }}
        .game-log-table td.game-log-fpts {{ font-weight: 700; background: rgba(45,212,191,0.15); }}
        .game-log-table thead tr:first-child th {{
          text-align: center; border-bottom: 1px solid rgba(128,128,128,0.4);
        }}
        .game-log-table thead tr:last-child th {{
          font-weight: 700; font-size: 0.75rem; letter-spacing: 0.02em;
          border-bottom: 2px solid rgba(128,128,128,0.5);
        }}
        .game-log-table tbody tr:nth-of-type(even) {{ background: rgba(128,128,128,0.06); }}
      </style>
      <table class="game-log-table">
        {thead}
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


# --- End Sleeper standard PPR scoring + per-game log -------------------------


@st.cache_data(show_spinner=False)
def load_stats(seasons_text: str):
    seasons = parse_seasons(seasons_text)
    df = load_recent_player_stats(seasons)
    return _enrich_with_schedule_details(df, seasons)


def _search_players(searchterm: str) -> list[str]:
    """Callback for st_searchbox: called on every keystroke (debounced)."""
    if not searchterm or len(searchterm.strip()) < 1:
        return []
    seasons_text = st.session_state.get("seasons_input") or ", ".join(
        str(season) for season in parse_seasons(None)
    )
    try:
        df = load_stats(seasons_text)
    except Exception:
        return []
    # Restrict player-name candidates to offensive skill positions (team-name
    # suggestions are unaffected, since every team fields offensive players).
    if "position" in df.columns:
        df = df.filter(pl.col("position").is_in(OFFENSIVE_POSITIONS))
    return get_autocomplete_suggestions(df, searchterm, limit=8)


def render_result(answer: dict[str, any]) -> None:
    if not answer:
        st.warning("No result returned.")
        return

    st.subheader(answer.get("title", "Result"))
    st.write(answer.get("summary", ""))

    projection = answer.get("projection")
    if projection:
        with st.expander("Projection details", expanded=True):
            st.metric("Projection", projection.get("projection"))
            st.metric("Recent average", projection.get("recent_average"))
            st.metric("Sample average", projection.get("sample_average"))
            st.metric("Trend", projection.get("direction"))
            st.write(projection.get("method"))

    game_log = answer.get("game_log")
    has_game_log = bool(game_log and game_log.get("rows"))

    rows = answer.get("rows", [])
    # For a matched player, the game log below already covers this data
    # (reordered, styled, with Sleeper PPR scoring), so skip the raw table.
    if rows and not has_game_log:
        st.dataframe(rows)
    elif not rows and not has_game_log:
        st.info("No rows available for this query.")

    if has_game_log:
        st.write("#### Game log (Sleeper standard PPR)")
        table_html = render_game_log_table(game_log)
        if table_html:
            st.markdown(table_html, unsafe_allow_html=True)

    splits = answer.get("splits")
    if splits and splits.get("sections"):
        st.write("#### Stat splits")
        table_html = render_splits_table(splits)
        if table_html:
            st.markdown(table_html, unsafe_allow_html=True)


def run_search():
    # Seasons and the search box live outside the form: widgets inside an
    # st.form only rerun the script on submit. st.text_input also only
    # commits on blur/Enter, not per keystroke, so it can't drive live
    # suggestions either. st_searchbox is a component built for exactly
    # this: it calls _search_players on every keystroke (debounced) and
    # renders a live dropdown, with no need to press Enter or tab away.
    default_seasons = list(parse_seasons(None))
    current_season = default_seasons[-1]
    season_options = list(range(current_season, 1998, -1))
    selected_seasons = st.multiselect(
        "Seasons",
        options=season_options,
        default=default_seasons,
        key="seasons_multiselect",
        help="Select one or more seasons",
    )
    seasons = ", ".join(str(season) for season in sorted(selected_seasons))
    # _search_players reads this key directly (it only gets a search term,
    # not the multiselect's value) to know which seasons to search within.
    st.session_state["seasons_input"] = seasons
    query = st_searchbox(
        _search_players,
        key="player_searchbox",
        placeholder="Player Search",
        default_use_searchterm=True,
        clear_on_submit=False,
    ) or ""

    with st.form("search_form"):
        left, right = st.columns([3, 2])
        with left:
            compare = st.text_input(
                "Compare players",
                help="Enter 2-3 player names separated by commas, and, or vs.",
            )
        with right:
            location = st.selectbox("Location", ["", "home", "away"])
            roof = st.selectbox("Roof", ["", "indoors", "outdoors"])
            day = st.selectbox(
                "Day", ["", "weekday", "weekend", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            )
            time_of_day = st.selectbox(
                "Time of day", ["", "europe", "early", "midday", "night"]
            )
            weather = st.selectbox("Weather", ["", "rain", "snow", "below_0"])
            divisional = st.checkbox("Divisional rivals only")

        submitted = st.form_submit_button("Search")

    if submitted:
        if not query and not compare:
            st.warning("Enter a query or compare players.")
            return

        try:
            df = load_stats(seasons)
        except Exception as exc:
            st.error(f"Failed to load stats: {exc}")
            return

        params = {
            "location": [location],
            "roof": [roof],
            "day": [day],
            "time_of_day": [time_of_day],
            "weather": [weather],
            "divisional": ["1" if divisional else ""],
        }

        filtered_df = _apply_split_filters(df, params)
        compare_names = parse_compare_names(compare)
        answer = answer_query(
            filtered_df,
            SearchConfig(
                query=query,
                seasons=parse_seasons(seasons),
                limit=12,
                compare=compare_names,
            ),
        )

        if answer.get("type") == "player":
            parsed_seasons = parse_seasons(seasons)
            splits = compute_player_splits(df, answer["title"])
            if splits:
                answer["splits"] = splits
            game_log = compute_game_log(df, answer["title"], parsed_seasons)
            if game_log:
                answer["game_log"] = game_log

        render_result(answer)


run_search()

st.markdown("---")

st.header("Sleeper league lookup")

username = st.text_input("Sleeper username for league lookup", value="")
season = st.selectbox("Sleeper season", [2026, 2025, 2024], index=0, key="sleeper_season")

if st.button("Find My Leagues", key="find_leagues"):
    if not username:
        st.warning("Enter your Sleeper username.")
    else:
        user_id = get_user_id(username)
        if isinstance(user_id, str) and user_id.startswith("Error"):
            st.error(user_id)
        else:
            st.success(f"User ID found: {user_id}")
            leagues = get_leagues(user_id, season)
            if isinstance(leagues, str) and leagues.startswith("Error"):
                st.error(leagues)
            elif not leagues:
                st.warning("No leagues found for this user and season.")
            else:
                st.session_state["leagues"] = leagues

if "leagues" in st.session_state:
    leagues = st.session_state["leagues"]
    selected_league_name = st.selectbox("Select a league", list(leagues.keys()), key="selected_league_name")
    selected_league_id = leagues[selected_league_name]
    st.write("Selected League ID:", selected_league_id)
    if st.button("View Rosters", key="view_rosters"):
        rosters = get_all_rosters(selected_league_id)
        if isinstance(rosters, str) and rosters.startswith("Error"):
            st.error(rosters)
        else:
            st.subheader("League Rosters")
            st.json(rosters)
