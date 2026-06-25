import nflreadpy as nfl
import pandas as pd

# Load data
player_stats = nfl.load_player_stats(list(range(2000, 2026))).to_pandas()
team_stats = nfl.load_team_stats(seasons=True).to_pandas()
schedules = nfl.load_schedules().to_pandas()
contracts = nfl.load_contracts().to_pandas()

player_stats.to_csv('player_stats.csv')
team_stats.to_csv('team_stats.csv')
schedules.to_csv('schedules.csv')
contracts.to_csv('contracts.csv')

player_contracts = player_stats.merge(
    contracts,
    left_on=['player_display_name', 'season'],
    right_on=['player', 'year_signed'],
    how="left",
    suffixes=("_player_stats", "_contract")
)

player_contracts.to_csv('player_contracts.csv')

main_df = player_contracts.merge(
    schedules,
    left_on=['season', 'week', 'team_player_stats'],
    right_on=['season', 'week', 'home_team'],
    how="left",
    suffixes=("_player_stats", "_home_team")
)

main_df = main_df.merge(
    schedules,
    left_on=['season', 'week', 'team_player_stats'],
    right_on=['season', 'week', 'away_team'],
    how="left",
    suffixes=("_player_stats", "_away_team")
)

main_df = main_df.to_csv('main_df.csv')

home_games = schedules[
    ["season", "home_team"]
].rename(columns={"home_team": "team"})

away_games = schedules[
    ["season", "away_team"]
].rename(columns={"away_team": "team"})

team_schedule = (
    pd.concat([home_games, away_games])
    .groupby(["season", "team"])
    .size()
    .reset_index(name="games_scheduled")
)


team_info = team_stats.merge(
    team_schedule,
    on=["season", "team"],
    how="left"
)





"""
#https://github.com/nflverse/nflreadpy
import nflreadpy as nfl
import pandas as pd

player_stats = nfl.load_player_stats(list(range(2000, 2026))).to_pandas()
team_stats = nfl.load_team_stats(seasons=True).to_pandas()
schedules =  nfl.load_schedules().to_pandas()
contracts = nfl.load_contracts().to_pandas()

#brady = player_stats[player_stats['player_id']=='00-0019596']
#brady.to_csv('output.csv', index=False)

#print(brady.head())
#team_stats.to_csv('team_output.csv', index=False)

#nfl.load_schedules().to_pandas().to_csv('schedules.csv')
#nfl.load_injuries().to_pandas().to_csv('injuries.csv')
#nfl.load_contracts().to_pandas().to_csv('contracts.csv')
#nfl.load_ff_rankings().to_pandas().to_csv('rankings.csv')
"""