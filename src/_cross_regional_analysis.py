"""Cross-regional matchup analysis — exploring signal for international prediction."""
import pandas as pd
import numpy as np
import glob
from collections import defaultdict

# Load ALL data
files = [f for f in glob.glob('data/csv/*.csv') if not f.endswith('.bak')]
print(f'Loading {len(files)} CSV files...')
df = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(files)], ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
print(f'Total rows: {len(df):,}  Date range: {df.date.min().date()} to {df.date.max().date()}')

# ── Tournament leagues ──
tournaments = {'MSI', 'WLDs', 'EWC', 'FST', 'DCup', 'ASI', 'Asia Master', 'CCWS', 'WCC', 'AC'}

team_rows = df[df['position'] == 'team'].copy()
domestic = team_rows[~team_rows['league'].isin(tournaments)].sort_values('date')

# Build team -> home league (last domestic appearance)
team_home = {}
for _, row in domestic.iterrows():
    team_home[row['teamid']] = row['league']

# Map domestic leagues -> major regions
league_to_region = {
    'LCK': 'LCK', 'LCKC': 'LCK',
    'LPL': 'LPL', 'LDL': 'LPL',
    'LEC': 'LEC', 'LFL': 'LEC', 'LFL2': 'LEC', 'PRM': 'LEC', 'NLC': 'LEC',
    'LVP SL': 'LEC', 'AL': 'LEC', 'EBL': 'LEC', 'EM': 'LEC', 'EUM': 'LEC',
    'HLL': 'LEC', 'GL': 'LEC', 'ESLOL': 'LEC', 'HM': 'LEC', 'GLL': 'LEC',
    'LIT': 'LEC', 'UL': 'LEC', 'EL': 'LEC', 'EPL': 'LEC',
    'LCS': 'LCS', 'NACL': 'LCS', 'LTA N': 'LCS',
    'LTA': 'LTA', 'LTA S': 'LTA',
    'CBLOL': 'CBLOL', 'CBLOLA': 'CBLOL', 'CD': 'CBLOL',
    'LLA': 'LLA', 'LAS': 'LLA',
    'PCS': 'PCS', 'PCL': 'PCS',
    'VCS': 'VCS', 'LJL': 'LJL', 'LCO': 'LCO', 'TCL': 'TCL', 'LCP': 'LCP',
}

# ── Extract cross-regional games ──
intl = team_rows[team_rows['league'].isin(tournaments)]
game_pairs = []
for gid, grp in intl.groupby('gameid'):
    if len(grp) != 2:
        continue
    rows = grp.sort_values('result', ascending=False)
    w, l = rows.iloc[0], rows.iloc[1]
    wr = league_to_region.get(team_home.get(w['teamid'], '?'), '?')
    lr = league_to_region.get(team_home.get(l['teamid'], '?'), '?')
    if wr == lr or wr == '?' or lr == '?':
        continue
    game_pairs.append({
        'date': w['date'], 'league': w['league'],
        'w_team': w['teamname'], 'l_team': l['teamname'],
        'w_region': wr, 'l_region': lr,
    })

gp = pd.DataFrame(game_pairs)
gp['date'] = pd.to_datetime(gp['date'])
gp['year'] = gp['date'].dt.year
print(f'\nCross-regional international games: {len(gp)}')
print(f'\nPer event:')
print(gp.groupby('league').size().sort_values(ascending=False).to_string())

# ── Build matchup stats ──
matchups = defaultdict(lambda: [0, 0])  # [wins, total_games]
for _, row in gp.iterrows():
    matchups[(row['w_region'], row['l_region'])][0] += 1
    matchups[(row['w_region'], row['l_region'])][1] += 1
    matchups[(row['l_region'], row['w_region'])][1] += 1

regions = ['LCK', 'LPL', 'LEC', 'LCS', 'PCS', 'VCS', 'CBLOL', 'LLA', 'LCP', 'TCL', 'LJL']

# ══════════════════════════════════════════════════════════════
# 1. RAW WIN RATE MATRIX
# ══════════════════════════════════════════════════════════════
print('\n' + '='*120)
print(' CROSS-REGIONAL WIN RATE MATRIX (2022-2026)')
print(' Row beats Column at win% (games played)')
print('='*120)

hdr = '         '
for r in regions:
    hdr += '{:>10s}'.format(r)
print(hdr)
print('-' * len(hdr))

for r1 in regions:
    line = '{:>8s} '.format(r1)
    for r2 in regions:
        if r1 == r2:
            line += '      ---  '
        else:
            w, t = matchups[(r1, r2)]
            if t > 0:
                pct = w / t * 100
                line += '  {:4.0f}%({:3d})'.format(pct, t)
            else:
                line += '         - '
    print(line)

# ══════════════════════════════════════════════════════════════
# 2. OVERALL INTERNATIONAL PERFORMANCE
# ══════════════════════════════════════════════════════════════
print('\n' + '='*70)
print(' OVERALL INTERNATIONAL PERFORMANCE BY REGION')
print('='*70)
region_stats = defaultdict(lambda: [0, 0])
for _, row in gp.iterrows():
    region_stats[row['w_region']][0] += 1
    region_stats[row['w_region']][1] += 1
    region_stats[row['l_region']][1] += 1

print('{:>8s}  {:>6s}  {:>6s}  {:>6s}'.format('Region', 'Wins', 'Games', 'Win%'))
print('-' * 32)
for r in sorted(region_stats.keys(), key=lambda x: region_stats[x][0]/max(region_stats[x][1],1), reverse=True):
    w, t = region_stats[r]
    print('{:>8s}  {:>6d}  {:>6d}  {:>5.1f}%'.format(r, w, t, w/t*100))

# ══════════════════════════════════════════════════════════════
# 3. YEAR-OVER-YEAR STABILITY (key question: is this persistent?)
# ══════════════════════════════════════════════════════════════
print('\n' + '='*90)
print(' YEAR-OVER-YEAR: TOP REGION PAIRS (stability check)')
print(' Does the LCK > LEC edge persist? Does LPL > LCS persist?')
print('='*90)

top_pairs = [('LCK', 'LEC'), ('LCK', 'LPL'), ('LPL', 'LEC'), ('LCK', 'LCS'),
             ('LPL', 'LCS'), ('LEC', 'LCS'), ('LCK', 'PCS'), ('LPL', 'PCS'),
             ('LEC', 'PCS'), ('LCK', 'VCS'), ('LPL', 'VCS')]

print('{:>12s}'.format('Matchup'), end='')
for y in sorted(gp['year'].unique()):
    print('  {:>12s}'.format(str(y)))
print('  {:>12s}'.format('ALL'))
print('-' * (14 + 14 * (len(gp['year'].unique()) + 1)))

for r1, r2 in top_pairs:
    label = '{} vs {}'.format(r1, r2)
    print('{:>12s}'.format(label), end='')
    for y in sorted(gp['year'].unique()):
        yr_data = gp[gp['year'] == y]
        w1 = len(yr_data[(yr_data['w_region'] == r1) & (yr_data['l_region'] == r2)])
        w2 = len(yr_data[(yr_data['w_region'] == r2) & (yr_data['l_region'] == r1)])
        total = w1 + w2
        if total > 0:
            pct = w1 / total * 100
            print('  {:4.0f}%({:3d})  '.format(pct, total), end='')
        else:
            print('           -  ', end='')
    # All years
    w1 = matchups[(r1, r2)][0]
    t = matchups[(r1, r2)][1]
    if t > 0:
        print('  {:4.0f}%({:3d})  '.format(w1/t*100, t), end='')
    print()

# ══════════════════════════════════════════════════════════════
# 4. THE KEY QUESTION: ELO residuals by region pair
# Now run ELO on all data, then check: when LCK plays LEC,
# does LCK outperform their ELO expectation?
# ══════════════════════════════════════════════════════════════
print('\n' + '='*90)
print(' ELO RESIDUAL ANALYSIS: Does region X outperform ELO vs region Y?')
print(' (Positive = row region wins MORE than ELO predicts vs column)')
print('='*90)

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from features.elo import PlayerEloSystem

# Run ELO on all data chronologically
elo = PlayerEloSystem()
players = df[df['position'] != 'team'][['gameid', 'teamid', 'playerid', 'position']].drop_duplicates()
team_data = df[df['position'] == 'team'][['gameid', 'teamid', 'teamname', 'date', 'league', 'result', 'side']].drop_duplicates()
team_data = team_data.sort_values('date')

# Group by game
game_elo_data = []
grouped = team_data.groupby('gameid')
players_grouped = players.groupby(['gameid', 'teamid'])['playerid'].apply(list).to_dict()

print('\nProcessing ELOs across all games...')
count = 0
for gid, grp in grouped:
    if len(grp) != 2:
        continue
    grp = grp.sort_values('side')  # Blue first
    t1, t2 = grp.iloc[0], grp.iloc[1]
    
    p1 = players_grouped.get((gid, t1['teamid']), [])
    p2 = players_grouped.get((gid, t2['teamid']), [])
    if len(p1) != 5 or len(p2) != 5:
        continue
    
    date = pd.to_datetime(t1['date'])
    league = t1['league']
    
    # Get pre-match ELOs
    avg_a = np.mean([elo.get_player_elo(pid, date, league) for pid in p1])
    avg_b = np.mean([elo.get_player_elo(pid, date, league) for pid in p2])
    expected_a = elo.calculate_expected_score(avg_a, avg_b)
    
    result = elo.process_match(date, league, p1, p2, t1['result'] == 1)
    
    if elo.is_tournament(league):
        r1_region = league_to_region.get(team_home.get(t1['teamid'], '?'), '?')
        r2_region = league_to_region.get(team_home.get(t2['teamid'], '?'), '?')
        if r1_region != '?' and r2_region != '?' and r1_region != r2_region:
            game_elo_data.append({
                'gameid': gid, 'date': date, 'league': league,
                'team_a': t1['teamname'], 'team_b': t2['teamname'],
                'region_a': r1_region, 'region_b': r2_region,
                'elo_a': avg_a, 'elo_b': avg_b,
                'expected_a': expected_a, 'actual_a': t1['result'],
            })
    count += 1
    if count % 10000 == 0:
        print(f'  Processed {count} games...')

print(f'  Done. {count} total games processed, {len(game_elo_data)} cross-regional.')

elo_df = pd.DataFrame(game_elo_data)
elo_df['residual_a'] = elo_df['actual_a'] - elo_df['expected_a']  # positive = A outperformed ELO

# Build residual matrix
print('\n{:>12s}'.format('vs'), end='')
for r in regions:
    print('{:>10s}'.format(r), end='')
print('{:>10s}'.format('ALL'))
print('-' * (12 + 10 * (len(regions) + 1)))

for r1 in regions:
    print('{:>12s}'.format(r1), end='')
    all_resids = []
    for r2 in regions:
        if r1 == r2:
            print('{:>10s}'.format('---'), end='')
            continue
        # Games where r1 is team_a
        mask_ab = (elo_df['region_a'] == r1) & (elo_df['region_b'] == r2)
        # Games where r1 is team_b (flip residual)
        mask_ba = (elo_df['region_a'] == r2) & (elo_df['region_b'] == r1)
        
        resids = list(elo_df[mask_ab]['residual_a']) + list(-elo_df[mask_ba]['residual_a'])
        n = len(resids)
        if n >= 3:
            avg = np.mean(resids) * 100  # as percentage points
            all_resids.extend(resids)
            print('{:>+5.1f}%({:3d})'.format(avg, n), end='')
        else:
            print('{:>10s}'.format('-'), end='')
    
    # Overall residual for this region
    # All games involving this region
    mask_a = elo_df['region_a'] == r1
    mask_b = elo_df['region_b'] == r1
    all_r = list(elo_df[mask_a]['residual_a']) + list(-elo_df[mask_b]['residual_a'])
    if all_r:
        print('{:>+5.1f}%({:3d})'.format(np.mean(all_r)*100, len(all_r)), end='')
    print()

print('\nPositive = region in row outperforms ELO expectation vs region in column')
print('e.g. +5.0%(50) means the row region wins 5% more often than ELO predicts across 50 games')
