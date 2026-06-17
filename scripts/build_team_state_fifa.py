#!/usr/bin/env python3
"""Build team_state_2026.csv with FIFA rankings included.

Generates the current state of each World Cup 2026 team:
  - Latest Elo rating (post-match from last played game)
  - Squad market value, caps, age (from convocatoria aggregation)
  - Latest FIFA ranking points (most recent ranking available)
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ELO_PATH = ROOT / 'datos' / 'master' / 'elo_por_partido.csv'
SQUAD_PATH = ROOT / 'datos' / 'master' / 'squad_features.csv'
RANKS_PATH = ROOT / 'datos' / 'historico de partidos' / 'FIFA World Rankings' / 'fifa_mens_rank.csv'
OUT_PATH = ROOT / 'datos' / 'master' / 'team_state_2026.csv'

# Load data
elo_df = pd.read_csv(ELO_PATH, parse_dates=['date'])
squad_df = pd.read_csv(SQUAD_PATH)
ranks_df = pd.read_csv(RANKS_PATH, parse_dates=['date'])

# Extract latest Elo for each team (post-match from most recent date)
elo_df_sorted = elo_df.sort_values('date')

# Home teams' latest Elo
home_elo = elo_df_sorted.groupby('home_team').last()[['elo_home_post']].rename(
    columns={'home_team': 'team', 'elo_home_post': 'elo'}
).reset_index()
home_elo.columns = ['team', 'elo']

# Away teams' latest Elo
away_elo = elo_df_sorted.groupby('away_team').last()[['elo_away_post']].rename(
    columns={'away_team': 'team', 'elo_away_post': 'elo'}
).reset_index()
away_elo.columns = ['team', 'elo']

# Combine and take the max (most recent)
latest_elo = pd.concat([home_elo, away_elo]).groupby('team')['elo'].max().reset_index()

# Get latest FIFA ranking for each team
latest_fifa = ranks_df.sort_values('date').groupby('team').last().reset_index()
latest_fifa = latest_fifa[['team', 'rank', 'total.points']].rename(
    columns={'rank': 'fifa_rank', 'total.points': 'fifa_points'}
)

# Merge: squad features (left join on squad_df) + Elo + FIFA ranks
state = squad_df.copy()
state = state.rename(columns={'canonical': 'team'})

# Add latest Elo
state = state.merge(latest_elo, on='team', how='left')

# Add latest FIFA ranking
state = state.merge(latest_fifa, on='team', how='left')

# Fill missing FIFA ranks with 0
state['fifa_rank'] = state['fifa_rank'].fillna(0)
state['fifa_points'] = state['fifa_points'].fillna(0)

state.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH}")
print(f"Teams: {len(state)}")
print(f"Sample:\n{state[['team', 'elo', 'fifa_rank', 'fifa_points']].head()}")
