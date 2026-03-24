"""Quick check on unclassified leagues."""
import pandas as pd, glob
files = [f for f in glob.glob('data/csv/*.csv') if not f.endswith('.bak')]
dfs = [pd.read_csv(f, low_memory=False) for f in sorted(files)]
df = pd.concat(dfs, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
team = df[df['position']=='team']

for league in ['IC', 'KeSPA', 'LES', 'CT', 'CCWS', 'EM', 'CD', 'EBL', 'HC', 'HLL', 'HW', 'LFL2', 'LIT', 'LPLOL', 'LRN', 'LRS', 'NEXO', 'NLC', 'RL', 'ROL', 'PRMP', 'HM', 'LAS']:
    lg = team[team['league']==league].sort_values('date')
    if len(lg) == 0:
        continue
    teams_sample = list(lg['teamname'].unique()[:5])
    first = str(lg.iloc[0]['date'])[:10]
    last = str(lg.iloc[-1]['date'])[:10]
    n_teams = lg['teamname'].nunique()
    print(f"{league:<8} {n_teams:>3} teams  {first} to {last}  e.g. {teams_sample}")
