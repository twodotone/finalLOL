import pandas as pd, glob, os

files = [f for f in glob.glob('data/csv/*.csv') if not f.endswith('.bak')]
raw = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
raw['date'] = pd.to_datetime(raw['date'])
team_rows = raw[raw['position']=='team']

for name in ['Nightbirds', 'BRUTE']:
    t = team_rows[team_rows['teamname']==name].sort_values('date')
    recent = t.tail(10)
    league = t['league'].iloc[-1]
    wr5 = t.tail(5)['result'].mean()*100
    wr10 = recent['result'].mean()*100
    last_date = t['date'].iloc[-1].date()
    
    print(f"\n=== {name} ({league}) ===")
    print(f"Last match: {last_date}")
    print(f"Last 5 WR: {wr5:.0f}%  |  Last 10 WR: {wr10:.0f}%")
    print(f"Last 5 results: {t.tail(5)['result'].tolist()}")
    print(f"Last 5 GD@15: {t.tail(5)['golddiffat15'].tolist()}")
    print(f"Last 10 results: {recent['result'].tolist()}")

    # Who did they play?
    last5 = t.tail(5)
    for _, g in last5.iterrows():
        gid = g['gameid']
        opp = team_rows[(team_rows['gameid']==gid) & (team_rows['teamname']!=name)]
        opp_name = opp['teamname'].iloc[0] if len(opp)>0 else '?'
        print(f"  {g['date'].date()} vs {opp_name}: {'W' if g['result'] else 'L'} (GD@15: {g['golddiffat15']:.0f})")

# Also check SOS-adjusted values from features CSV
df = pd.read_csv('data/processed/model_features_v2.csv')
nb_id = t.iloc[0]['teamid']  # will be BRUTE's id from last loop
# Re-get IDs properly
for name in ['Nightbirds', 'BRUTE']:
    t = team_rows[team_rows['teamname']==name]
    tid = t['teamid'].iloc[-1]
    feats = df[df['teamid']==tid]
    last = feats.iloc[-1]
    print(f"\n--- {name} Feature Row ---")
    print(f"  team_elo_pre: {last.get('team_elo_pre', 'N/A'):.1f}")
    print(f"  opp_elo_pre: {last.get('opp_elo_pre', 'N/A'):.1f}")
    for prefix in ['roll5', 'roll10']:
        gd = last.get(f'{prefix}_adj_golddiffat15', 0)
        dpm = last.get(f'{prefix}_adj_dpm', 0)
        sos = last.get(f'{prefix}_opp_elo_pre', 0)
        print(f"  {prefix}: GD@15={gd:.0f}, DPM={dpm:.0f}, SOS={sos:.0f}")
