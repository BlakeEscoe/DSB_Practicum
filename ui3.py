import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CAPSTONE_DIR = ROOT_DIR / "Capstone"

for path in (str(CAPSTONE_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

_lineups = None
_trades = None


def _get_capstone_modules():
    global _lineups, _trades

    if _lineups is not None and _trades is not None:
        return _lineups, _trades

    try:
        import lineups
        import trades
    except Exception as exc:
        raise ImportError(
            f"Unable to import required Capstone modules from {CAPSTONE_DIR}: {exc}"
        )

    _lineups = lineups
    _trades = trades
    return _lineups, _trades


def optimize_starting_lineups(league_id):
    """Build optimized starting lineups for every roster in the league."""
    lineups, _ = _get_capstone_modules()
    return lineups.optimize_starting_lineups(league_id)


def get_best_trades_by_position(league_id, team_id, position):
    """Return candidate trades to acquire a starter at the requested position."""
    lineups, trades = _get_capstone_modules()
    optimized_lineups = lineups.optimize_starting_lineups(league_id)
    return trades.get_best_trades_by_position(
        league_id,
        team_id,
        position,
        optimized_lineups,
    )


def get_trades_to_improve_both_starting_lineups(league_id, team_id, min_gain=0.1):
    """Return trades that improve both teams' optimized starting lineups."""
    lineups, trades = _get_capstone_modules()
    optimized_lineups = lineups.optimize_starting_lineups(league_id)
    return trades.get_trades_to_improve_both_starting_lineups(
        league_id,
        team_id,
        optimized_lineups,
        min_gain=min_gain,
    )


def run_trade_optimizer():
    import streamlit as st

    st.subheader("Trade Optimizer")
    st.write(
        "Run your Sleeper league trade optimizer using optimized starting lineups and candidate trade packages."
    )

    league_id = st.text_input("Sleeper league ID", value="", key="trade_league_id")
    team_id_input = st.text_input("Your roster ID", value="", key="trade_team_id")
    position = st.selectbox("Target position", ["QB", "RB", "WR", "TE"], key="trade_position")
    mode = st.radio(
        "Optimizer mode",
        ["Optimized starting lineups", "Best trades by position", "Mutually beneficial trades"],
        horizontal=True,
        key="trade_optimizer_mode",
    )
    min_gain = st.slider(
        "Minimum gain for both teams",
        min_value=0.0,
        max_value=5.0,
        value=0.1,
        step=0.1,
        key="trade_optimizer_min_gain",
    )

    if st.button("Run optimizer", key="run_trade_optimizer"):
        if not league_id:
            st.error("Enter a Sleeper league ID.")
            return

        try:
            team_id = int(team_id_input)
        except ValueError:
            st.error("Your roster ID must be an integer.")
            return

        with st.spinner("Computing trade results..."):
            try:
                if mode == "Optimized starting lineups":
                    optimized_lineups = optimize_starting_lineups(league_id)
                    if optimized_lineups.empty:
                        st.warning("No optimized lineup data returned.")
                    else:
                        st.success("Optimized starting lineups computed.")
                        st.dataframe(optimized_lineups, use_container_width=True)
                elif mode == "Best trades by position":
                    trades_df = get_best_trades_by_position(league_id, team_id, position)
                    if trades_df.empty:
                        st.warning("No candidate trades found for this position.")
                    else:
                        st.success("Candidate trades found.")
                        st.dataframe(trades_df, use_container_width=True)
                else:
                    trades_df = get_trades_to_improve_both_starting_lineups(
                        league_id,
                        team_id,
                        min_gain=min_gain,
                    )
                    if trades_df.empty:
                        st.warning("No mutually beneficial trades were found.")
                    else:
                        st.success("Mutually beneficial trades found.")
                        st.dataframe(trades_df, use_container_width=True)
            except Exception as exc:
                st.error(f"Unable to compute trade optimizer results: {exc}")


if __name__ == "__main__":
    try:
        import streamlit as st

        run_trade_optimizer()
    except ImportError:
        print("Use `streamlit run ui3.py` to launch the trade optimizer UI.")
