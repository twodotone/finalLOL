"""Temp script: Analyze FURIA's ELO trajectory and ranking divergence."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import glob
import os
from cli_predictor import build_elo_engine
from features.elo import PlayerEloSystem

# ======== PART 0: ELO TRAJECTORY GAME-BY-GAME ========
files0 = [f for f in glob.glob('data/csv/*.csv') if not f.endswith('.bak')]
dfs0 = [pd.read_csv(f, low_memory=False) for f in sorted(files0)]
df0 = pd.concat(dfs0, ignore_index=True)
df0['date'] = pd.to_datetime(df0['date'])
df0 = df0.sort_values('date').reset_index(drop=True)
team0 = df0[df0['position'] == 'team']
players0 = df0[df0['position'] != 'team'][['gameid', 'teamid', 'playerid']]
grouped0 = players0.groupby(['gameid', 'teamid'])['playerid'].apply(list).to_dict()

elo0 = PlayerEloSystem()
furia_traj = []
for _, grp in team0.groupby('gameid', sort=False):
    if len(grp) != 2:
        continue
    date = grp['date'].iloc[0]
    league = grp['league'].iloc[0]
    a = grp.iloc[0]
    b = grp.iloc[1]
    a_won = bool(a['result'])
    ka = (a['gameid'], a['teamid'])
    kb = (a['gameid'], b['teamid'])
    if ka not in grouped0 or kb not in grouped0:
        continue
    pa = grouped0[ka]
    pb = grouped0[kb]
    if len(pa) != 5 or len(pb) != 5:
        continue
    result = elo0.process_match(date, league, pa, pb, a_won)
    if a['teamname'] == 'FURIA' or b['teamname'] == 'FURIA':
        is_a = a['teamname'] == 'FURIA'
        furia_players = pa if is_a else pb
        opp_name = b['teamname'] if is_a else a['teamname']
        won = a_won if is_a else not a_won
        furia_avg = np.mean([elo0.players[pid]['elo'] for pid in furia_players])
        furia_traj.append({
            'date': date, 'league': league, 'opp': opp_name,
            'won': won, 'furia_elo': furia_avg,
            'cross_region': result['is_cross_region'],
        })

tdf = pd.DataFrame(furia_traj)
print("=== FURIA ELO TRAJECTORY (last 25 games) ===")
for _, r in tdf.tail(25).iterrows():
    cr = ' [INTL K=40]' if r['cross_region'] else ''
    w = 'W' if r['won'] else 'L'
    dt = str(r['date'])[:10]
    print(f"  {dt}  {r['league']:>12}  vs {r['opp']:<20}  {w}  ELO: {r['furia_elo']:.0f}{cr}")

pre_ac = tdf[tdf['date'] < '2026-03-01'].iloc[-1]['furia_elo']
post_ac = tdf.iloc[-1]['furia_elo']
ac_games = tdf[(tdf['date'] >= '2026-03-01') & (tdf['league'] == 'Americas Cup')]
ac_gain = post_ac - pre_ac
print(f"\nPre-Americas Cup ELO: {pre_ac:.0f}")
print(f"Post-Americas Cup ELO: {post_ac:.0f}")
print(f"ELO gained from Americas Cup: +{ac_gain:.0f} ({len(ac_games)} games, all wins)")
print(f"Average gain per win: +{ac_gain / max(len(ac_games), 1):.1f}")
print()
# Now skip the rest of the slow part — we already have elo_engine from build_elo_engine below

# Load data
files = [f for f in glob.glob('data/csv/*.csv') if not f.endswith('.bak')]
dfs = [pd.read_csv(f, low_memory=False) for f in sorted(files)]
df = pd.concat(dfs, ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
team = df[df['position'] == 'team']

# Build ELO engine
elo_engine, bridge, match_results, players_df = build_elo_engine('data/csv')

today = pd.Timestamp.now()

# === FURIA match history ===
furia = team[team['teamname'] == 'FURIA'].sort_values('date')
print("=== FURIA LAST 20 GAMES ===")
for _, r in furia.tail(20).iterrows():
    gid = r['gameid']
    opp_rows = team[(team['gameid'] == gid) & (team['teamname'] != 'FURIA')]
    if len(opp_rows):
        opp = opp_rows.iloc[0]
        w = 'W' if r['result'] == 1 else 'L'
        print(f"  {str(r['date'])[:10]}  {r['league']:>10}  vs {opp['teamname']:<25}  {w}")

# === 2026 breakdown by league ===
print("\n=== FURIA 2026 BY LEAGUE ===")
f26 = furia[furia['date'] >= '2026-01-01']
for league in f26['league'].unique():
    lg = f26[f26['league'] == league]
    wins = int(lg['result'].sum())
    losses = len(lg) - wins
    print(f"\n  {league}: {wins}W-{losses}L")
    for _, r in lg.iterrows():
        gid = r['gameid']
        opp_rows = team[(team['gameid'] == gid) & (team['teamname'] != 'FURIA')]
        if len(opp_rows):
            opp = opp_rows.iloc[0]
            w = 'W' if r['result'] == 1 else 'L'
            print(f"    {str(r['date'])[:10]}  vs {opp['teamname']:<25}  {w}")

# === Cross-region K factor analysis ===
print("\n=== ELO SYSTEM K-FACTOR ANALYSIS ===")
print(f"  Domestic K: {elo_engine.base_k}")
print(f"  Cross-region K: {elo_engine.base_k * elo_engine.intl_k_multiplier}")
print(f"  Cross-region multiplier: {elo_engine.intl_k_multiplier:.1f}x")

# === What ELO does the model assign FURIA's opponents? ===
print("\n=== FURIA'S AMERICAS CUP OPPONENTS ELO CHECK ===")
ac_teams = ['Cloud9', 'Sentinels', 'LYON', 'Leviatán', 'Isurus', 'paiN Gaming', 'LOUD', 'RED Canids']
for tname in ac_teams:
    t_rows = match_results[match_results['teamname'] == tname]
    if len(t_rows) == 0:
        # Try partial match
        t_rows = match_results[match_results['teamname'].str.contains(tname, case=False, na=False)]
    if len(t_rows) == 0:
        print(f"  {tname}: NOT FOUND")
        continue
    tid = t_rows.iloc[-1]['teamid']
    last_game = t_rows.iloc[-1]['gameid']
    league = t_rows.iloc[-1]['league']
    roster = players_df[(players_df['teamid'] == tid) & (players_df['gameid'] == last_game)]['playerid'].tolist()
    if len(roster) == 5:
        elos = [elo_engine.get_player_elo(pid, today, league) for pid in roster]
        avg = sum(elos) / 5
        print(f"  {tname:<25}  ELO: {avg:.0f}  League: {league}  Last: {str(t_rows.iloc[-1]['date'])[:10]}")
    else:
        print(f"  {tname:<25}  (roster != 5)")

# === Regional bridge adjustments ===
print("\n=== REGIONAL BRIDGE ADJUSTMENTS ===")
regions = ['Americas', 'EMEA', 'Asia', 'China']
for r1 in regions:
    for r2 in regions:
        if r1 < r2:
            adj = bridge.get_bridge_adj(today, r1, r2)
            if abs(adj) > 0.001:
                print(f"  {r1} vs {r2}: {adj:+.4f}")

# === Compare: filter to active major-league teams only ===
print("\n=== CLEANED TOP 20 (active 2026 teams, major leagues only) ===")
major = ['LCK', 'LPL', 'LEC', 'LCS', 'LTA', 'CBLOL', 'PCS', 'VCS', 'LJL']
# Also include teams that played in international events
latest = team.sort_values('date').groupby('teamid').last().reset_index()
# Filter to teams active in 2026
active = latest[latest['date'] >= '2026-01-01']

rows = []
for _, t in active.iterrows():
    tid = t['teamid']
    tname = t['teamname']
    league = t['league']
    last_game = t['gameid']
    
    domestic = league
    if elo_engine.is_tournament(league):
        team_hist = match_results[match_results['teamid'] == tid]
        for _, row in team_hist.iloc[::-1].iterrows():
            if not elo_engine.is_tournament(row['league']):
                domestic = row['league']
                break
    
    roster = players_df[(players_df['teamid'] == tid) & (players_df['gameid'] == last_game)]['playerid'].tolist()
    if len(roster) != 5:
        continue
    elos = [elo_engine.get_player_elo(pid, today, domestic) for pid in roster]
    avg = sum(elos) / 5
    rows.append({'team': tname, 'league': domestic, 'elo': avg})

rdf = pd.DataFrame(rows).sort_values('elo', ascending=False).head(25)
rdf['rank'] = range(1, len(rdf) + 1)
for _, r in rdf.iterrows():
    print(f"  {r['rank']:>3}. {r['team']:<25}  {r['league']:>6}  {r['elo']:.0f}")
