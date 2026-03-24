"""
Audit: find teams from Riot GPR that are missing or mislinked in our data.
Also analyzes all leagues in our dataset for tier calibration.
"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import glob
from collections import Counter
from features.elo import PlayerEloSystem

files = [f for f in glob.glob('data/csv/*.csv') if not f.endswith('.bak')]
dfs = [pd.read_csv(f, low_memory=False) for f in sorted(files)]
df = pd.concat(dfs, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
team = df[df['position'] == 'team']

# ====================================
# PART 1: MISSING TEAM AUDIT
# ====================================
print("=" * 70)
print("PART 1: MISSING TEAM AUDIT — Riot GPR teams we can't find")
print("=" * 70)

missing_names = [
    'CTBC Flying Oyster', 'FlyQuest', 'Invictus Gaming', 
    'Deep Cross Gaming', 'Cloud9', 'GAM Esports',
    'Team Liquid', 'Nongshim', 'MVK Esports'
]

for name in missing_names:
    # Try exact match
    exact = team[team['teamname'] == name]
    # Try partial/contains
    partial = team[team['teamname'].str.contains(name.split()[0], case=False, na=False)]
    
    print(f"\n--- {name} ---")
    if len(exact) > 0:
        last = exact.sort_values('date').iloc[-1]
        print(f"  FOUND (exact): last game {str(last['date'])[:10]}, league={last['league']}, teamid={last['teamid']}")
    else:
        # Show partial matches
        candidates = partial['teamname'].unique()[:10]
        if len(candidates) > 0:
            print(f"  NOT FOUND (exact). Partial matches:")
            for c in candidates:
                c_rows = team[team['teamname'] == c].sort_values('date')
                last = c_rows.iloc[-1]
                games_2026 = len(c_rows[c_rows['date'] >= '2026-01-01'])
                print(f"    '{c}' — last: {str(last['date'])[:10]}, league: {last['league']}, 2026 games: {games_2026}")
        else:
            print(f"  NOT FOUND at all")

# Special deep dives
print("\n--- SPECIAL: FlyQuest / LCS team search ---")
lcs_2026 = team[(team['league'] == 'LCS') & (team['date'] >= '2026-01-01')]
lcs_teams = lcs_2026.groupby('teamname').agg(
    games=('gameid', 'nunique'),
    last_date=('date', 'max')
).sort_values('games', ascending=False)
print("  All LCS teams in 2026:")
for tname, row in lcs_teams.iterrows():
    print(f"    {tname:<30} games: {row['games']:>3}  last: {str(row['last_date'])[:10]}")

print("\n--- SPECIAL: LTA / Americas teams in 2026 ---")
lta_2026 = team[(team['league'].str.contains('LTA|Americas|LCS', na=False)) & (team['date'] >= '2026-01-01')]
lta_teams = lta_2026.groupby(['teamname', 'league']).agg(
    games=('gameid', 'nunique'),
    last_date=('date', 'max')
).sort_values('games', ascending=False)
print("  All LTA/LCS/Americas teams in 2026:")
for (tname, league), row in lta_teams.iterrows():
    print(f"    {tname:<30} {league:<15} games: {row['games']:>3}  last: {str(row['last_date'])[:10]}")

print("\n--- SPECIAL: LCP teams in 2026 ---")
lcp_2026 = team[(team['league'] == 'LCP') & (team['date'] >= '2026-01-01')]
lcp_teams = lcp_2026.groupby('teamname').agg(
    games=('gameid', 'nunique'),
    last_date=('date', 'max')
).sort_values('games', ascending=False)
print("  All LCP teams in 2026:")
for tname, row in lcp_teams.iterrows():
    print(f"    {tname:<30} games: {row['games']:>3}  last: {str(row['last_date'])[:10]}")

# ====================================
# PART 2: LEAGUE TIER ANALYSIS
# ====================================
print("\n" + "=" * 70)
print("PART 2: ALL LEAGUES IN OUR DATA (2025-2026) — for tier calibration")
print("=" * 70)

recent = team[team['date'] >= '2025-01-01']
league_stats = recent.groupby('league').agg(
    total_games=('gameid', 'nunique'),
    teams=('teamname', 'nunique'),
    first_date=('date', 'min'),
    last_date=('date', 'max')
).sort_values('total_games', ascending=False)

print(f"\n{'LEAGUE':<15} {'GAMES':>6} {'TEAMS':>6} {'FIRST':>12} {'LAST':>12}")
print("-" * 60)
for league, row in league_stats.iterrows():
    print(f"  {league:<13} {row['total_games']:>6} {row['teams']:>6} {str(row['first_date'])[:10]:>12} {str(row['last_date'])[:10]:>12}")

# ====================================
# PART 3: What are all distinct tournament_leagues in our data?
# ====================================
print("\n" + "=" * 70)
print("PART 3: ALL TOURNAMENT/INTERNATIONAL LEAGUES IN OUR DATA")
print("=" * 70)

known_tournaments = {'MSI', 'WCC', 'EWC', 'WLDs', 'FST', 'DCup', 'ASI', 'Asia Master', 'CCWS', 'AC', 'Americas Cup'}
all_leagues = team['league'].unique()
intl_candidates = [l for l in all_leagues if any(kw in l.lower() for kw in ['cup', 'world', 'master', 'intl', 'international', 'msi', 'ewc', 'asia'])]
intl_candidates += [l for l in all_leagues if l in known_tournaments]
intl_candidates = sorted(set(intl_candidates))

print("  Leagues matching international keywords:")
for l in intl_candidates:
    cnt = len(team[team['league'] == l])
    last = team[team['league'] == l]['date'].max()
    in_set = "YES" if l in known_tournaments else "NO"
    print(f"    {l:<25} games: {cnt:>5}  last: {str(last)[:10]}  in tournament_set: {in_set}")

# Check for leagues NOT in any tier and NOT in tournaments
print("\n  Leagues NOT in any tier or tournament set:")
elo_sys = PlayerEloSystem()
all_classified = set(elo_sys.league_base_elo.keys()) | elo_sys.tournament_leagues
unclassified = [l for l in all_leagues if l not in all_classified]
for l in sorted(unclassified):
    cnt = len(team[(team['league'] == l) & (team['date'] >= '2025-01-01')])
    if cnt > 0:
        print(f"    {l:<25} 2025+ games: {cnt}")
