#!/usr/bin/env python3
"""Merge FIFA men's rankings into the master dataset.

Creates `datos/master/dataset_final_fifa.csv` with added columns:
  - home_fifa_rank, home_fifa_points
  - away_fifa_rank, away_fifa_points

Matching strategy: for each match date we take the latest FIFA ranking entry
for that team with date <= match date (backward fill / merge_asof per team).
If no prior ranking exists, values will be NaN.
"""
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_IN = ROOT / 'datos' / 'master' / 'dataset_final.csv'
RANKS_IN = ROOT / 'datos' / 'historico de partidos' / 'FIFA World Rankings' / 'fifa_mens_rank.csv'
DATA_OUT = ROOT / 'datos' / 'master' / 'dataset_final_fifa.csv'


def load_data():
    df = pd.read_csv(DATA_IN, parse_dates=['date'])
    ranks = pd.read_csv(RANKS_IN, parse_dates=['date'])
    return df, ranks


def normalize_names(df, col):
    return df[col].astype(str).str.strip().str.lower()


def merge_for_side(df, ranks, side):
    # side: 'home' or 'away'
    df_side = df[['date', f'{side}_team']].copy()
    df_side['team_norm'] = normalize_names(df_side, f'{side}_team')
    ranks_sorted = ranks[['date', 'team', 'acronym', 'rank', 'total.points']].copy()
    ranks_sorted['team_norm'] = ranks_sorted['team'].astype(str).str.strip().str.lower()

    results = []
    # perform group-wise merge_asof per team to avoid global sort issues
    for team in df_side['team_norm'].unique():
        left_sub = df_side[df_side['team_norm'] == team].sort_values('date')
        right_sub = ranks_sorted[ranks_sorted['team_norm'] == team].sort_values('date')

        if right_sub.empty:
            # no ranking available for this team: fill NaNs
            res_sub = pd.DataFrame({
                'fifa_rank': [pd.NA] * len(left_sub),
                'fifa_points': [pd.NA] * len(left_sub)
            }, index=left_sub.index)
        else:
            right_sub = right_sub.rename(columns={'date': 'rank_date', 'rank': 'fifa_rank', 'total.points': 'fifa_points'})
            merged_sub = pd.merge_asof(
                left_sub.sort_values('date'),
                right_sub[['rank_date', 'team_norm', 'fifa_rank', 'fifa_points']].sort_values('rank_date'),
                left_on='date',
                right_on='rank_date',
                direction='backward',
            )
            res_sub = merged_sub[['fifa_rank', 'fifa_points']]

        results.append(res_sub)

    merged = pd.concat(results).sort_index()
    return merged[['fifa_rank', 'fifa_points']].reset_index(drop=True)


def main():
    df, ranks = load_data()

    print(f'Rows in dataset: {len(df):,}')
    print(f'Rows in ranks: {len(ranks):,}')

    home = merge_for_side(df, ranks, 'home')
    away = merge_for_side(df, ranks, 'away')

    df_out = df.copy()
    df_out['home_fifa_rank'] = home['fifa_rank']
    df_out['home_fifa_points'] = home['fifa_points']
    df_out['away_fifa_rank'] = away['fifa_rank']
    df_out['away_fifa_points'] = away['fifa_points']

    df_out.to_csv(DATA_OUT, index=False)
    filled_home = df_out['home_fifa_rank'].notna().sum()
    filled_away = df_out['away_fifa_rank'].notna().sum()
    print(f'Wrote {DATA_OUT} — home ranks filled: {filled_home:,}, away ranks filled: {filled_away:,}')


if __name__ == '__main__':
    main()
