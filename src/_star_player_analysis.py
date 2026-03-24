"""How often does the team with the highest-rated individual player win?"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/model_features_v2.csv')
lane_cols = ['lane_elo_top','lane_elo_jng','lane_elo_mid','lane_elo_bot','lane_elo_sup']

df['max_player_elo'] = df[lane_cols].max(axis=1)
df['star_pos'] = df[lane_cols].idxmax(axis=1).str.replace('lane_elo_','')

# Filter to games where lane ELOs have diverged from base
valid = df[df['max_player_elo'] != 1500].copy()
n_games = valid['gameid'].nunique()
print(f"Total rows: {len(df)}, Rows with non-base ELO: {len(valid)}")
print(f"Unique games with non-base ELO: {n_games}")

# Pair up teams per game
games = valid.groupby('gameid').filter(lambda g: len(g) == 2)
rows = []
for gid, g in games.groupby('gameid'):
    if len(g) != 2:
        continue
    t1, t2 = g.iloc[0], g.iloc[1]
    rows.append({
        'gameid': gid,
        'team1_max': t1['max_player_elo'],
        'team2_max': t2['max_player_elo'],
        'team1_result': t1['result'],
        'team2_result': t2['result'],
        'team1_star': t1['star_pos'],
        'team2_star': t2['star_pos'],
        'team1_elo': t1['team_elo_pre'],
        'team2_elo': t2['team_elo_pre'],
    })
gp = pd.DataFrame(rows)

# Which team has the highest-rated individual player?
gp['star_team_wins'] = np.where(
    gp['team1_max'] > gp['team2_max'], gp['team1_result'],
    np.where(gp['team2_max'] > gp['team1_max'], gp['team2_result'], np.nan)
)

# Also check: does the better TEAM elo win?
gp['better_team_wins'] = np.where(
    gp['team1_elo'] > gp['team2_elo'], gp['team1_result'],
    np.where(gp['team2_elo'] > gp['team1_elo'], gp['team2_result'], np.nan)
)

# Drop ties
gp_star = gp.dropna(subset=['star_team_wins'])
gp_team = gp.dropna(subset=['better_team_wins'])
gp_star['elo_gap'] = abs(gp_star['team1_max'] - gp_star['team2_max'])

star_wr = gp_star['star_team_wins'].mean()
team_wr = gp_team['better_team_wins'].mean()

print(f"\n{'='*60}")
print(f" STAR PLAYER ANALYSIS")
print(f"{'='*60}")
print(f" Games analysed: {len(gp_star)}")
print(f" Team with highest ELO player wins: {star_wr:.1%}  ({int(gp_star['star_team_wins'].sum())}/{len(gp_star)})")
print(f" Team with higher TEAM ELO wins:    {team_wr:.1%}  ({int(gp_team['better_team_wins'].sum())}/{len(gp_team)})")

print(f"\n By star-player ELO gap:")
for lo, hi, label in [(0,50,'0-50'),(50,100,'50-100'),(100,200,'100-200'),(200,400,'200-400'),(400,9999,'400+')]:
    sub = gp_star[(gp_star['elo_gap'] >= lo) & (gp_star['elo_gap'] < hi)]
    if len(sub) > 0:
        print(f"   Gap {label:>7}: {sub['star_team_wins'].mean():.1%}  ({len(sub)} games)")

# Does the star player position matter?
print(f"\n By which POSITION is the star player:")
for pos in ['top','jng','mid','bot','sup']:
    # games where team1's star is in this position AND team1 has the higher max
    mask1 = (gp_star['team1_max'] > gp_star['team2_max']) & (gp_star['team1_star'] == pos)
    mask2 = (gp_star['team2_max'] > gp_star['team1_max']) & (gp_star['team2_star'] == pos)
    wins = gp_star.loc[mask1, 'team1_result'].sum() + gp_star.loc[mask2, 'team2_result'].sum()
    total = mask1.sum() + mask2.sum()
    if total > 0:
        print(f"   Star is {pos:>3}: {wins/total:.1%}  ({int(wins)}/{total} games)")

# Interesting: when star player team != better team ELO, who wins?
print(f"\n CONFLICT: Star player is on the WEAKER team (by team ELO):")
conflict = gp_star.copy()
conflict['star_is_team1'] = conflict['team1_max'] > conflict['team2_max']
conflict['team1_is_better'] = conflict['team1_elo'] > conflict['team2_elo']
disagree = conflict[conflict['star_is_team1'] != conflict['team1_is_better']]
if len(disagree) > 0:
    # In these games, the star player's team is the underdog by team ELO
    disagree_wins = np.where(
        disagree['star_is_team1'], disagree['team1_result'], disagree['team2_result']
    ).astype(float).mean()
    print(f"   {len(disagree)} games where star player is on weaker team")
    print(f"   Star player's (weaker) team wins: {disagree_wins:.1%}")
    print(f"   (vs 50% baseline — star player advantage: {disagree_wins - 0.5:+.1%})")
