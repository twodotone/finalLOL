"""
build_series_outcomes.py

Groups Oracle Elixir individual game rows into series, labels each series as
a sweep (fav won without dropping a game) or competitive, then joins pre-game-1
features from model_features_v2.csv to build a training dataset.

Output: data/processed/series_outcomes.csv
"""

import pandas as pd
import numpy as np
import glob
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'csv')
FEATURES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'model_features_v2.csv')
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'series_outcomes.csv')


def load_raw():
    files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, '*.csv')) if not f.endswith('.bak')])
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


def extract_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups individual game rows into series.
    A series = consecutive games between the same two teams where 'game' resets to 1.
    We use gameid + game number: gameid for game N in a series shares the same
    base ID pattern (e.g. LOLTMNT03_185025 for all games in that match).

    Oracle Elixir structure:
      - 'game' column = game number within series (1, 2, 3 ...)
      - Same gameid can appear for different game numbers depending on the feed.
      - For structured feeds (e.g. _game_1, _game_2), gameid has suffix.
      - For unstructured feeds (LOLTMNT style), we group by (teamid, opp_teamid, date trunc).

    Strategy:
      1. Filter team rows only (position == 'team')  
      2. For each unique gameid, extract the two teams, date, game number, result
      3. Group by (team_a_id, team_b_id, series_key) where series_key is derived
         from the gameid base (strip _game_N suffix) or from date+team pair
      4. Determine series winner and number of games played
    """
    team_rows = df[df['position'] == 'team'].copy()
    team_rows = team_rows.dropna(subset=['gameid', 'teamid', 'game', 'result', 'date'])
    team_rows['game'] = pd.to_numeric(team_rows['game'], errors='coerce')
    team_rows = team_rows.dropna(subset=['game'])
    team_rows['game'] = team_rows['game'].astype(int)

    # Build game-level summary: one row per game per team
    games = team_rows[['gameid', 'game', 'date', 'league', 'teamid', 'teamname', 'result']].copy()
    games['result'] = pd.to_numeric(games['result'], errors='coerce').fillna(0).astype(int)

    # Derive series key from gameid: strip _game_N suffix if present
    def series_key(gameid: str) -> str:
        import re
        return re.sub(r'_game_\d+$', '', str(gameid))

    games['series_key'] = games['gameid'].map(series_key)

    # For each series_key, pair up the two teams per game number
    # Pivot so each row is one game with team_a and team_b
    grouped = games.groupby(['series_key', 'game'])
    game_pairs = []
    for (sk, gnum), grp in grouped:
        if len(grp) != 2:
            continue  # skip malformed
        a, b = grp.iloc[0], grp.iloc[1]
        game_pairs.append({
            'series_key': sk,
            'game': gnum,
            'date': a['date'],
            'league': a['league'],
            'team_a_id': a['teamid'],
            'team_a_name': a['teamname'],
            'team_b_id': b['teamid'],
            'team_b_name': b['teamname'],
            'team_a_won': int(a['result']),
            'first_gameid': a['gameid'],
        })

    pairs_df = pd.DataFrame(game_pairs)
    if pairs_df.empty:
        raise ValueError("No game pairs found — check data structure")

    # Sort pairs so team_a/team_b are consistent within a series
    # Canonicalize: always put lower teamid as team_a
    mask = pairs_df['team_a_id'] > pairs_df['team_b_id']
    pairs_df.loc[mask, ['team_a_id', 'team_b_id']] = (
        pairs_df.loc[mask, ['team_b_id', 'team_a_id']].values
    )
    pairs_df.loc[mask, ['team_a_name', 'team_b_name']] = (
        pairs_df.loc[mask, ['team_b_name', 'team_a_name']].values
    )
    pairs_df.loc[mask, 'team_a_won'] = 1 - pairs_df.loc[mask, 'team_a_won']

    # Now group by (series_key, team_a_id, team_b_id) to get full series
    series_records = []
    for (sk, ta, tb), grp in pairs_df.groupby(['series_key', 'team_a_id', 'team_b_id']):
        grp = grp.sort_values('game')
        games_played = grp['game'].max()
        if games_played < 2:
            continue  # skip BO1s

        team_a_wins = grp['team_a_won'].sum()
        team_b_wins = games_played - team_a_wins

        # Determine series winner
        if team_a_wins > team_b_wins:
            winner_id = ta
            winner_wins = team_a_wins
        elif team_b_wins > team_a_wins:
            winner_id = tb
            winner_wins = team_b_wins
        else:
            continue  # incomplete / tied series

        # Sweep = winner won all games without losing any
        is_sweep = int(winner_wins == games_played)

        # Series format
        if games_played <= 3:
            fmt = 'BO3'
        elif games_played <= 5:
            fmt = 'BO5'
        else:
            continue  # unexpected

        # Game 1 gameid for feature joins
        game1_rows = grp[grp['game'] == 1]
        if game1_rows.empty:
            continue
        game1_gameid = game1_rows.iloc[0]['first_gameid']

        series_records.append({
            'series_key': sk,
            'date': grp['date'].min(),
            'league': grp['league'].iloc[0],
            'format': fmt,
            'team_a_id': ta,
            'team_b_id': tb,
            'team_a_name': grp['team_a_name'].iloc[0],
            'team_b_name': grp['team_b_name'].iloc[0],
            'games_played': games_played,
            'team_a_wins': int(team_a_wins),
            'team_b_wins': int(team_b_wins),
            'winner_id': winner_id,
            'is_sweep': is_sweep,
            'game1_gameid': game1_gameid,
        })

    series_df = pd.DataFrame(series_records)
    print(f"Total series found: {len(series_df)}")
    print(f"  Sweeps: {series_df['is_sweep'].sum()} ({series_df['is_sweep'].mean()*100:.1f}%)")
    print(f"  Competitive: {(~series_df['is_sweep'].astype(bool)).sum()}")
    print(f"  BO3: {(series_df['format']=='BO3').sum()}")
    print(f"  BO5: {(series_df['format']=='BO5').sum()}")
    return series_df


def join_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join model_features_v2.csv to get pre-series features.
    We take the features from game 1 (the first game in the series) as the
    'pre-series snapshot' for each team.
    """
    print("\nLoading feature matrix...")
    feats = pd.read_csv(FEATURES_PATH)

    # Key feature columns we want
    feature_cols = [
        'gameid', 'teamid', 'side',
        'team_elo_pre', 'opp_elo_pre', 'expected_win_prob',
        # 5-game rolling stats
        'roll5_adj_golddiffat15', 'roll5_adj_xpdiffat15', 'roll5_adj_csdiffat15',
        'roll5_adj_firstblood', 'roll5_adj_firstdragon', 'roll5_adj_firstherald',
        'roll5_adj_firsttower', 'roll5_adj_firstbaron',
        'roll5_adj_dpm', 'roll5_adj_vspm',
        # 10-game rolling
        'roll10_adj_golddiffat15', 'roll10_adj_dpm', 'roll10_adj_vspm',
        # Lane ELOs
        'lane_elo_top', 'lane_elo_jng', 'lane_elo_mid', 'lane_elo_bot', 'lane_elo_sup',
        'lane_delta_avg', 'lane_delta_worst',
        # Cross-league
        'min_cross_league_games', 'delta_cross_league_games',
    ]
    # Keep only cols that exist
    feature_cols = [c for c in feature_cols if c in feats.columns]
    feats = feats[feature_cols].copy()

    # Join team_a features
    a_feats = feats.copy()
    a_feats.columns = [f'a_{c}' if c not in ('gameid', 'teamid', 'side') else c for c in a_feats.columns]

    # Join team_b features (same gameid, other teamid)
    b_feats = feats.copy()
    b_feats.columns = [f'b_{c}' if c not in ('gameid', 'teamid', 'side') else c for c in b_feats.columns]

    # Merge: match game1_gameid + team_a_id
    joined = series_df.merge(
        a_feats.rename(columns={'gameid': 'game1_gameid', 'teamid': 'team_a_id'}),
        on=['game1_gameid', 'team_a_id'],
        how='left'
    )
    joined = joined.merge(
        b_feats.rename(columns={'gameid': 'game1_gameid', 'teamid': 'team_b_id'}),
        on=['game1_gameid', 'team_b_id'],
        how='left'
    )

    before = len(joined)
    joined = joined.dropna(subset=['a_team_elo_pre', 'b_team_elo_pre'])
    after = len(joined)
    print(f"After feature join: {after}/{before} series have complete features")

    # Build delta features (favorite's perspective: winner vs loser)
    # We'll frame it as: fav = whoever the model says is the favorite (higher ELO)
    joined['fav_is_a'] = (joined['a_team_elo_pre'] >= joined['b_team_elo_pre']).astype(int)

    def diff_col(col_a, col_b, fav_is_a):
        """Returns fav - underdog for a stat column."""
        return np.where(fav_is_a == 1, col_a - col_b, col_b - col_a)

    # ELO gap (always positive, fav - underdog)
    joined['elo_gap'] = np.abs(joined['a_team_elo_pre'] - joined['b_team_elo_pre'])
    joined['win_prob_fav'] = np.where(
        joined['fav_is_a'] == 1,
        joined['a_expected_win_prob'],
        joined['b_expected_win_prob']
    )

    stat_pairs = [
        ('roll5_adj_golddiffat15', 'delta_gd15_5'),
        ('roll5_adj_dpm', 'delta_dpm_5'),
        ('roll5_adj_vspm', 'delta_vspm_5'),
        ('roll5_adj_firstblood', 'delta_fb_5'),
        ('roll5_adj_firstdragon', 'delta_fd_5'),
        ('roll5_adj_firstherald', 'delta_fh_5'),
        ('roll5_adj_firsttower', 'delta_ft_5'),
        ('roll5_adj_firstbaron', 'delta_fbaron_5'),
        ('roll10_adj_golddiffat15', 'delta_gd15_10'),
        ('roll10_adj_dpm', 'delta_dpm_10'),
    ]
    for (raw_col, delta_name) in stat_pairs:
        a_col = f'a_{raw_col}'
        b_col = f'b_{raw_col}'
        if a_col in joined.columns and b_col in joined.columns:
            joined[delta_name] = diff_col(
                joined[a_col].fillna(0).values,
                joined[b_col].fillna(0).values,
                joined['fav_is_a'].values
            )

    # Lane features
    for pos in ['top', 'jng', 'mid', 'bot', 'sup']:
        a_col = f'a_lane_elo_{pos}'
        b_col = f'b_lane_elo_{pos}'
        if a_col in joined.columns and b_col in joined.columns:
            joined[f'lane_gap_{pos}'] = diff_col(
                joined[a_col].fillna(1500).values,
                joined[b_col].fillna(1500).values,
                joined['fav_is_a'].values
            )

    if 'a_lane_delta_avg' in joined.columns:
        joined['fav_lane_avg'] = np.where(
            joined['fav_is_a'] == 1,
            joined['a_lane_delta_avg'].fillna(0),
            -joined['b_lane_delta_avg'].fillna(0)
        )
    if 'a_lane_delta_worst' in joined.columns:
        joined['fav_lane_worst'] = np.where(
            joined['fav_is_a'] == 1,
            joined['a_lane_delta_worst'].fillna(0),
            joined['b_lane_delta_worst'].fillna(0)
        )

    # Absolute DPM for both (high DPM = high variance = competitive)
    if 'a_roll5_adj_dpm' in joined.columns:
        joined['avg_dpm_both'] = (
            joined['a_roll5_adj_dpm'].fillna(0) + joined['b_roll5_adj_dpm'].fillna(0)
        ) / 2

    # Number of lanes fav wins
    lane_cols = [f'lane_gap_{pos}' for pos in ['top', 'jng', 'mid', 'bot', 'sup'] if f'lane_gap_{pos}' in joined.columns]
    if lane_cols:
        joined['fav_lanes_won'] = (joined[lane_cols] > 0).sum(axis=1)

    return joined.sort_values('date').reset_index(drop=True)


def main():
    print("Loading raw Oracle Elixir data...")
    df = load_raw()
    print(f"Loaded {len(df):,} rows across all years")

    series_df = extract_series(df)
    full_df = join_features(series_df)

    print(f"\nFinal dataset shape: {full_df.shape}")
    print(f"Date range: {full_df['date'].min().date()} to {full_df['date'].max().date()}")
    print(f"Sweep rate overall: {full_df['is_sweep'].mean()*100:.1f}%")

    # Show sweep rate by format
    for fmt in ['BO3', 'BO5']:
        sub = full_df[full_df['format'] == fmt]
        print(f"  {fmt} sweep rate: {sub['is_sweep'].mean()*100:.1f}% ({len(sub)} series)")

    full_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == '__main__':
    main()
