#!/usr/bin/env python3
"""Build team_state_2026.csv with FIFA rankings and recent form included.

Generates the current state of each World Cup 2026 team:
  - Latest Elo rating (post-match from last played game)
    - Squad market value, caps, age (from convocatoria aggregation)
    - Days since last match and recent form over the last 5 matches
  - Latest FIFA ranking points (most recent ranking available)
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

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

# Build recent form and days since last match from the same match history
home_history = elo_df_sorted[['date', 'home_team', 'home_score', 'away_score']].copy()
home_history = home_history.rename(columns={
    'home_team': 'team',
    'home_score': 'gf',
    'away_score': 'ga',
})
home_history['points'] = home_history.apply(
    lambda r: 3 if r['gf'] > r['ga'] else 1 if r['gf'] == r['ga'] else 0,
    axis=1,
)

away_history = elo_df_sorted[['date', 'away_team', 'home_score', 'away_score']].copy()
away_history = away_history.rename(columns={
    'away_team': 'team',
    'away_score': 'gf',
    'home_score': 'ga',
})
away_history['points'] = away_history.apply(
    lambda r: 3 if r['gf'] > r['ga'] else 1 if r['gf'] == r['ga'] else 0,
    axis=1,
)

history = pd.concat([home_history, away_history], ignore_index=True).sort_values('date')
reference_date = elo_df_sorted['date'].max()

recent_form_rows = []
for team, group in history.groupby('team'):
    last5 = group.tail(5)
    recent_form_rows.append({
        'team': team,
        'recent_form_5': round(last5['points'].mean(), 2),
        'last_match_date': group['date'].max(),
    })

recent_form_df = pd.DataFrame(recent_form_rows)
recent_form_df['days_since_last_match'] = (reference_date - recent_form_df['last_match_date']).dt.days

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

# Add recent form and last-match recency
state = state.merge(recent_form_df[['team', 'recent_form_5', 'days_since_last_match']], on='team', how='left')

# Fill missing FIFA ranks with 0
state['fifa_rank'] = state['fifa_rank'].fillna(0)
state['fifa_points'] = state['fifa_points'].fillna(0)
state['recent_form_5'] = state['recent_form_5'].fillna(0)
state['days_since_last_match'] = state['days_since_last_match'].fillna(0)

state.to_csv(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH}")
print(f"Teams: {len(state)}")
print(f"Sample:\n{state[['team', 'elo', 'recent_form_5', 'days_since_last_match', 'fifa_rank', 'fifa_points']].head()}")
