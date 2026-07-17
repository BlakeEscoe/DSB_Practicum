import html
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st
from streamlit_searchbox import st_searchbox

def _flatten_html(markup: str) -> str:
    """Strip leading whitespace from every line of an HTML fragment.

    st.markdown(..., unsafe_allow_html=True) still runs the text through a
    CommonMark parser first, and a line indented 4+ spaces reads as a
    fenced code block there - which silently turns these indented f-string
    templates into literal text instead of rendered HTML.
    """
    return "\n".join(line.lstrip() for line in markup.strip("\n").splitlines())


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
from sleeper_connect import (
    get_user_id,
    get_leagues,
    get_all_rosters,
    get_user_name,
    nfl_player_ids,
)
import pandas as pd
from pathlib import Path

try:
    from nflreadpy.ui import (
        SearchConfig,
        answer_query,
        get_autocomplete_suggestions,
        load_recent_player_stats,
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

st.set_page_config(page_title="Fantasy Football Stat Explorer", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_player_lookup():
    # This CSV is useful when we have a local player lookup file.
    # If it is missing, return an empty dictionary so the app can keep running
    # and use the Sleeper API lookup as a fallback.
    lookup_path = DATA_DIR / "main_df_with_sleeper_ids.csv"
    if not lookup_path.exists():
        return {}

    players_df = pd.read_csv(
        lookup_path,
        usecols=[
            "sleeper_id",
            "player_display_name",
            "position_player_stats",
            "team_player_stats",
        ],
    )
    players_df["sleeper_id"] = (
        pd.to_numeric(players_df["sleeper_id"], errors="coerce")
        .astype("Int64")
        .astype("string")
    )

    return (
        players_df.dropna(subset=["sleeper_id"])
        .drop_duplicates("sleeper_id")
        .set_index("sleeper_id")
        .to_dict("index")
    )


@st.cache_data
def cached_user_name(user_id):
    return get_user_name(user_id)


@st.cache_data
def load_sleeper_player_lookup():
    sleeper_players = nfl_player_ids()

    if isinstance(sleeper_players, str) and sleeper_players.startswith("Error"):
        return {}

    return {
        str(player_id): {
            "player_display_name": player_info.get("full_name", "Unknown"),
            "position_player_stats": player_info.get("position", ""),
            "team_player_stats": player_info.get("team", ""),
        }
        for player_id, player_info in sleeper_players.items()
    }


def build_roster_df(roster):
    player_lookup = load_player_lookup()
    sleeper_player_lookup = load_sleeper_player_lookup()
    starter_ids = {
        str(sleeper_id)
        for sleeper_id in roster.get("starters", [])
        if sleeper_id is not None
    }
    players = []

    for sleeper_id in roster.get("players", []) or []:
        sleeper_id_str = str(sleeper_id)
        lineup_status = "Starter" if sleeper_id_str in starter_ids else "Bench"

        # Handle Team Defense (DST)
        if sleeper_id_str.isalpha():
            players.append(
                {
                    "Lineup": lineup_status,
                    "Player": f"{sleeper_id_str} Defense",
                    "Position": "DEF",
                    "Team": sleeper_id_str,
                    "Sleeper ID": sleeper_id_str,
                }
            )

        # Handle regular players
        else:
            info = player_lookup.get(sleeper_id_str) or sleeper_player_lookup.get(
                sleeper_id_str, {}
            )

            players.append(
                {
                    "Lineup": lineup_status,
                    "Player": info.get("player_display_name", "Unknown"),
                    "Position": info.get("position_player_stats", ""),
                    "Team": info.get("team_player_stats", ""),
                    "Sleeper ID": sleeper_id_str,
                }
            )

    roster_df = pd.DataFrame(players)

    if roster_df.empty:
        return roster_df

    roster_df["Lineup Order"] = roster_df["Lineup"].map({"Starter": 0, "Bench": 1})

    return (
        roster_df.sort_values(["Lineup Order", "Position", "Player"])
        .drop(columns="Lineup Order")
        .reset_index(drop=True)
    )


def show_roster(roster):
    owner = roster.get("owner_id", "Unknown")
    owner_name = cached_user_name(owner) if owner != "Unknown" else "Unknown"
    st.caption(f"Fantasy General Manager: {owner_name}")

    roster_df = build_roster_df(roster)

    if roster_df.empty:
        st.info("No roster players found.")
        return

    starter_df = roster_df[roster_df["Lineup"] == "Starter"].drop(columns="Lineup")
    bench_df = roster_df[roster_df["Lineup"] == "Bench"].drop(columns="Lineup")

    with st.expander(f"Active Roster ({len(starter_df)})", expanded=True):
        st.dataframe(
            starter_df,
            hide_index=True,
            use_container_width=True,
        )

    with st.expander(f"Bench Players ({len(bench_df)})"):
        if bench_df.empty:
            st.caption("No bench players found.")
        else:
            st.dataframe(
                bench_df,
                hide_index=True,
                use_container_width=True,
            )


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

    return _flatten_html(f"""
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
    """)


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


# --- Side-by-side player comparison cards ------------------------------------
#
# Renders the compare-players search result as photo/team/position cards plus
# a stat table, in the spirit of FantasyPros' player-compare tool. We only
# have historical box-score stats (no upcoming matchup, odds, injury status,
# or expert consensus), so this covers season fantasy points and
# position-appropriate per-game stat totals, reusing the same stat-group
# helpers the ESPN-style splits table above uses.

_LOWER_IS_BETTER_METRICS = {"INT", "SACK"}


def compute_comparison(df: pl.DataFrame, player_names: tuple[str, ...]) -> dict[str, Any] | None:
    name_column = next(
        (c for c in ("player_display_name", "player_name", "display_name", "name") if c in df.columns),
        None,
    )
    if name_column is None:
        return None

    team_col = "recent_team" if "recent_team" in df.columns else "team" if "team" in df.columns else None
    sort_cols = [c for c in ("season", "week") if c in df.columns]

    players: list[dict[str, Any]] = []
    for name in player_names:
        rows = df.filter(pl.col(name_column) == name)
        if len(rows) == 0:
            continue
        if sort_cols:
            rows = rows.sort(sort_cols)

        position = None
        if "position" in rows.columns:
            positions = [str(p) for p in rows.select("position").to_series().to_list() if p]
            if positions:
                position = Counter(positions).most_common(1)[0][0]

        team = None
        if team_col:
            teams = [str(t) for t in rows.select(team_col).drop_nulls().to_series().to_list()]
            if teams:
                team = teams[-1]

        headshot = None
        if "headshot_url" in rows.columns:
            urls = [u for u in rows.select("headshot_url").drop_nulls().to_series().to_list() if u]
            headshot = urls[0] if urls else None

        games = len(rows)
        available_fpts = [c for c in _FPTS_COLUMNS if c in rows.columns]
        fpts_total = 0.0
        if available_fpts:
            sums = rows.select([pl.sum(c).alias(c) for c in available_fpts]).to_dicts()[0]
            fpts_total = round(_sleeper_ppr_points(sums), 2)
        fpts_avg = round(fpts_total / games, 2) if games else 0.0

        totals_cols = {c for _, cols in STAT_GROUP_BUILDERS.values() for c in cols}
        available_totals = [c for c in totals_cols if c in rows.columns]
        totals = (
            rows.select([pl.sum(c).alias(c) for c in available_totals]).to_dicts()[0]
            if available_totals
            else {}
        )
        groups = _player_stat_groups(position, totals)
        group_rows = {group: _aggregate_group_row(rows, group) for group in groups}

        players.append(
            {
                "name": name,
                "position": position,
                "team": team,
                "headshot": headshot,
                "games": games,
                "fpts_total": fpts_total,
                "fpts_avg": fpts_avg,
                "groups": groups,
                "group_rows": group_rows,
            }
        )

    if len(players) < 2:
        return None

    all_groups = [g for g in ("passing", "rushing", "receiving") if any(g in p["groups"] for p in players)]
    return {"players": players, "groups": all_groups}


def _best_indices(values: list[float], lower_is_better: bool = False) -> set[int]:
    if not values:
        return set()
    target = min(values) if lower_is_better else max(values)
    return {i for i, v in enumerate(values) if v == target}


def render_comparison(comparison: dict[str, Any]) -> str:
    players = comparison.get("players", [])
    groups = comparison.get("groups", [])
    if len(players) < 2:
        return ""

    n = len(players)

    header_cards = []
    for p in players:
        photo = (
            f'<img src="{html.escape(p["headshot"])}" alt="{html.escape(p["name"])}">'
            if p.get("headshot")
            else '<div class="compare-photo-placeholder"></div>'
        )
        team_position = " - ".join(x for x in (p.get("position"), p.get("team")) if x)
        header_cards.append(
            f"""
            <div class="compare-card">
              {photo}
              <div class="compare-name">{html.escape(p["name"])}</div>
              <div class="compare-meta">{html.escape(team_position)}</div>
              <div class="compare-meta">{p["games"]} games</div>
            </div>
            """
        )

    def stat_row(label: str, values: list[float], lower_is_better: bool = False) -> str:
        best = _best_indices(values, lower_is_better)
        cells = "".join(
            f'<td class="{"compare-best" if i in best else ""}">{html.escape(str(v))}</td>'
            for i, v in enumerate(values)
        )
        return f'<tr><td class="compare-label-col">{html.escape(label)}</td>{cells}</tr>'

    fpts_rows = stat_row("Season Total", [p["fpts_total"] for p in players]) + stat_row(
        "Season Avg", [p["fpts_avg"] for p in players]
    )

    group_sections = []
    for group in groups:
        headers = STAT_GROUP_HEADERS.get(group, [])
        rows_html = []
        for header in headers:
            values = [p["group_rows"].get(group, {}).get(header, 0) for p in players]
            rows_html.append(stat_row(header, values, lower_is_better=header in _LOWER_IS_BETTER_METRICS))
        group_sections.append(
            f"""
            <tr class="compare-section-row"><td colspan="{n + 1}">{html.escape(STAT_GROUP_LABELS.get(group, group.upper()))}</td></tr>
            {''.join(rows_html)}
            """
        )

    return _flatten_html(f"""
    <div class="compare-wrapper">
      <style>
        .compare-wrapper {{ margin: 0.5rem 0 1rem; }}
        .compare-cards {{
          display: grid; grid-template-columns: repeat({n}, 1fr); gap: 12px; margin-bottom: 12px;
        }}
        .compare-card {{
          text-align: center; padding: 12px; border-radius: 10px; background: rgba(128,128,128,0.08);
        }}
        .compare-card img, .compare-photo-placeholder {{
          width: 72px; height: 72px; border-radius: 50%; object-fit: cover; margin: 0 auto 8px;
          background: rgba(128,128,128,0.2);
        }}
        .compare-name {{ font-weight: 700; font-size: 1rem; }}
        .compare-meta {{ font-size: 0.8rem; opacity: 0.75; }}
        .compare-table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
        .compare-table td {{
          padding: 6px 10px; border-bottom: 1px solid rgba(128,128,128,0.25); text-align: center;
        }}
        .compare-table td.compare-label-col {{ text-align: left; font-weight: 600; }}
        .compare-table td.compare-best {{ font-weight: 700; background: rgba(45,212,191,0.18); }}
        .compare-section-row td {{
          text-align: left; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.04em;
          background: rgba(128,128,128,0.14); padding-top: 8px; padding-bottom: 8px; font-weight: 700;
        }}
      </style>
      <div class="compare-cards">{''.join(header_cards)}</div>
      <table class="compare-table">
        <tbody>
          {fpts_rows}
          {''.join(group_sections)}
        </tbody>
      </table>
    </div>
    """)


# --- End player comparison cards ---------------------------------------------

# Single "type of split" dropdown driving the game log filter, replacing the
# old set of 5 independent selectboxes (Location/Roof/Day/Time/Weather) that
# could be combined into an over-restrictive AND filter. Each option maps to
# exactly one (_apply_split_filters param, value) pair.
SPLIT_OPTIONS: dict[str, tuple[str, str]] = {
    "All games": ("", ""),
    "Location: Home": ("location", "home"),
    "Location: Away": ("location", "away"),
    "Roof: Indoors": ("roof", "indoors"),
    "Roof: Outdoors": ("roof", "outdoors"),
    "Day: Weekday": ("day", "weekday"),
    "Day: Weekend": ("day", "weekend"),
    "Day: Monday": ("day", "monday"),
    "Day: Tuesday": ("day", "tuesday"),
    "Day: Wednesday": ("day", "wednesday"),
    "Day: Thursday": ("day", "thursday"),
    "Day: Friday": ("day", "friday"),
    "Day: Saturday": ("day", "saturday"),
    "Day: Sunday": ("day", "sunday"),
    "Time of day: Europe": ("time_of_day", "europe"),
    "Time of day: Early": ("time_of_day", "early"),
    "Time of day: Midday": ("time_of_day", "midday"),
    "Time of day: Night": ("time_of_day", "night"),
    "Weather: Rain": ("weather", "rain"),
    "Weather: Snow": ("weather", "snow"),
    "Weather: Below 0°": ("weather", "below_0"),
    "Divisional rivals only": ("divisional", "1"),
}


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

    return _flatten_html(f"""
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
    """)


# --- End Sleeper standard PPR scoring + per-game log -------------------------


@st.cache_data(show_spinner=False)
def load_stats(seasons_text: str):
    seasons = parse_seasons(seasons_text)
    df = load_recent_player_stats(seasons)
    return _enrich_with_schedule_details(df, seasons)


def _make_search_players(seasons_state_key: str):
    """Build a st_searchbox callback bound to one tab's own seasons picker.

    Each tab (Stat Search / Compare Players) has its own seasons multiselect,
    so the live-suggest callback needs to read the matching session_state key
    rather than a single shared one.
    """

    def _search_players(searchterm: str) -> list[str]:
        if not searchterm or len(searchterm.strip()) < 1:
            return []
        seasons_text = st.session_state.get(seasons_state_key) or ", ".join(
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

    return _search_players


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

    comparison = answer.get("comparison")
    if comparison:
        table_html = render_comparison(comparison)
        if table_html:
            st.markdown(table_html, unsafe_allow_html=True)

    rows = answer.get("rows", [])
    # For a matched player, the game log below already covers this data
    # (reordered, styled, with Sleeper PPR scoring), so skip the raw table.
    # A compare result is rendered as cards above instead of a raw table too.
    if rows and not has_game_log and not comparison:
        st.dataframe(rows)
    elif not rows and not has_game_log and not comparison:
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


def _seasons_multiselect(key: str, state_key: str) -> str:
    default_seasons = list(parse_seasons(None))
    current_season = default_seasons[-1]
    season_options = list(range(current_season, 1998, -1))
    selected_seasons = st.multiselect(
        "Seasons",
        options=season_options,
        default=default_seasons,
        key=key,
        help="Select one or more seasons",
    )
    seasons = ", ".join(str(season) for season in sorted(selected_seasons))
    st.session_state[state_key] = seasons
    return seasons


def _split_selectbox(key: str) -> tuple[str, str, str]:
    split_label = st.selectbox(
        "Split",
        list(SPLIT_OPTIONS.keys()),
        key=key,
        help="Pick a split type to filter the game log to just those games.",
    )
    split_key, split_value = SPLIT_OPTIONS[split_label]
    return split_label, split_key, split_value


def _build_split_params(split_key: str, split_value: str) -> dict[str, list[str]]:
    params = {
        "location": [""],
        "roof": [""],
        "day": [""],
        "time_of_day": [""],
        "weather": [""],
        "divisional": [""],
    }
    if split_key:
        params[split_key] = [split_value]
    return params


def run_stat_search():
    # Seasons and the search box live outside the form: widgets inside an
    # st.form only rerun the script on submit. st.text_input also only
    # commits on blur/Enter, not per keystroke, so it can't drive live
    # suggestions either. st_searchbox is a component built for exactly
    # this: it calls the search callback on every keystroke (debounced) and
    # renders a live dropdown, with no need to press Enter or tab away.
    seasons = _seasons_multiselect("stat_seasons_multiselect", "stat_seasons_input")
    query = st_searchbox(
        _make_search_players("stat_seasons_input"),
        key="player_searchbox",
        placeholder="Player Search",
        default_use_searchterm=True,
        clear_on_submit=False,
    ) or ""

    with st.form("stat_search_form"):
        split_label, split_key, split_value = _split_selectbox("stat_split_select")
        submitted = st.form_submit_button("Search")

    if submitted:
        if not query:
            st.warning("Enter a query.")
            return

        try:
            df = load_stats(seasons)
        except Exception as exc:
            st.error(f"Failed to load stats: {exc}")
            return

        filtered_df = _apply_split_filters(df, _build_split_params(split_key, split_value))
        answer = answer_query(
            filtered_df,
            SearchConfig(query=query, seasons=parse_seasons(seasons), limit=12),
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


def run_compare():
    seasons = _seasons_multiselect("compare_seasons_multiselect", "compare_seasons_input")

    st.write("Compare players")
    st.caption("Select at least 2 players below.")
    compare_cols = st.columns(3)
    compare_selections: list[str] = []
    search_players = _make_search_players("compare_seasons_input")
    for i, col in enumerate(compare_cols):
        with col:
            selection = st_searchbox(
                search_players,
                key=f"compare_searchbox_{i}",
                placeholder=f"Player {i + 1}",
                default_use_searchterm=True,
                clear_on_submit=False,
            ) or ""
            compare_selections.append(selection)
    # Autocomplete (same live-suggest widget as the main search) guarantees
    # every compare slot is a name that actually exists in the data, instead
    # of a free-typed name with a typo silently dropping out of the compare.
    compare_names = tuple(name for name in compare_selections if name)

    with st.form("compare_form"):
        split_label, split_key, split_value = _split_selectbox("compare_split_select")
        submitted = st.form_submit_button("Compare")

    if submitted:
        if not compare_names:
            st.warning("Select at least 2 players to compare.")
            return

        if len(compare_names) < 2:
            st.warning(
                "Only one compare box has a selection. Pick a player from the "
                "dropdown in at least 2 of the 3 compare boxes to see a comparison."
            )
            return

        try:
            df = load_stats(seasons)
        except Exception as exc:
            st.error(f"Failed to load stats: {exc}")
            return

        filtered_df = _apply_split_filters(df, _build_split_params(split_key, split_value))
        answer = answer_query(
            filtered_df,
            SearchConfig(query="", seasons=parse_seasons(seasons), limit=12, compare=compare_names),
        )

        if answer.get("type") != "compare":
            missing = f" under the '{split_label}' split" if split_key else ""
            st.warning(
                f"Couldn't build a comparison for {', '.join(compare_names)}{missing} "
                "- one or more of them may have no games in the selected seasons."
            )
            return

        comparison = compute_comparison(filtered_df, compare_names)
        if comparison:
            answer["comparison"] = comparison

        render_result(answer)


def run_sleeper():
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
                st.session_state["user_id"] = user_id
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
                st.session_state["rosters"] = rosters

    if "rosters" in st.session_state:
        rosters = st.session_state["rosters"]
        user_id = st.session_state.get("user_id")

        st.subheader("League Rosters")

        user_roster = None
        other_rosters = []

        for roster in rosters:
            if roster.get("owner_id") == user_id:
                user_roster = roster
            else:
                other_rosters.append(roster)

        if user_roster:
            st.markdown("## Your Team")
            show_roster(user_roster)
            st.divider()

        for roster in other_rosters:
            owner = roster.get("owner_id", "Unknown")
            owner_name = cached_user_name(owner) if owner != "Unknown" else "Unknown"

            with st.expander(f"Team {owner_name}"):
                show_roster(roster)


tab_search, tab_compare, tab_sleeper = st.tabs(["Stat Search", "Compare Players", "Sleeper"])

with tab_search:
    run_stat_search()

with tab_compare:
    run_compare()

with tab_sleeper:
    run_sleeper()
