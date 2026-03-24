"""Post-fix validation: regenerate rankings and compare to Riot GPR."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from cli_predictor import build_elo_engine

elo_engine, bridge, match_results, players_df = build_elo_engine('data/csv')
today = pd.Timestamp.now()

team_data = match_results[['teamname','teamid','league','date']].drop_duplicates()
latest = team_data.sort_values('date').groupby('teamid').last().reset_index()
active = latest[latest['date'] >= '2026-01-01']

rows = []
for _, t in active.iterrows():
    tid = t['teamid']
    tname = t['teamname']
    league = t['league']
    last_game = t['gameid'] if 'gameid' in t.index else match_results[match_results['teamid']==tid].iloc[-1]['gameid']
    
    # Get latest gameid from match_results
    last_game = match_results[match_results['teamid']==tid].iloc[-1]['gameid']
    
    domestic = league
    if elo_engine.is_tournament(league):
        team_hist = match_results[match_results['teamid']==tid]
        for _, row in team_hist.iloc[::-1].iterrows():
            if not elo_engine.is_tournament(row['league']):
                domestic = row['league']
                break
    
    roster = players_df[(players_df['teamid']==tid) & (players_df['gameid']==last_game)]['playerid'].tolist()
    if len(roster) != 5:
        continue
    elos = [elo_engine.get_player_elo(pid, today, domestic) for pid in roster]
    avg = sum(elos)/5
    rows.append({'team': tname, 'league': domestic, 'elo': avg})

rdf = pd.DataFrame(rows).sort_values('elo', ascending=False).head(45)
rdf['rank'] = range(1, len(rdf)+1)

# Riot GPR for comparison
riot = {
    'Gen.G Esports': 1, 'T1': 2, 'Hanwha Life Esports': 3,
    'Bilibili Gaming': 4, 'KT Rolster': 5, "Anyone's Legend": 6,
    'G2 Esports': 7, 'CTBC Flying Oyster': 8, 'JD Gaming': 9,
    'Top Esports': 10, 'FlyQuest': 11, 'Weibo Gaming': 12,
    'BNK FEARX': 13, 'Dplus Kia': 13, 'Karmine Corp': 15,
    'Team Secret Whales': 16, 'Invictus Gaming': 17,
    'Deep Cross Gaming': 17, 'Cloud9': 19, 'Movistar KOI': 20,
    'GAM Esports': 21, 'Team Liquid': 22, 'LYON': 27,
    'DN SOOPers': 30, 'Edward Gaming': 29, 'FURIA': 38,
    'LNG Esports': 25, 'Nongshim RedForce': 26,
}

# Name matching
name_map = {
    "Anyone's Legend": "Anyone's Legend",
    'Nongshim RedForce': 'Nongshim RedForce',
}

print("=" * 90)
print(f"{'#':>3} {'TEAM':<28} {'LEAGUE':>6} {'ELO':>7}  {'RIOT':>5}  {'DIFF':>5}")
print("=" * 90)

for _, r in rdf.iterrows():
    tname = r['team']
    our_rank = r['rank']
    riot_rank = riot.get(tname)
    if riot_rank:
        diff = our_rank - riot_rank
        sign = '+' if diff > 0 else ''
        flag = ' **' if abs(diff) >= 7 else ''
        print(f"  {our_rank:>2}. {tname:<28} {r['league']:>6} {r['elo']:>7.0f}  R#{riot_rank:>2}  {sign}{diff:>4}{flag}")
    else:
        print(f"  {our_rank:>2}. {tname:<28} {r['league']:>6} {r['elo']:>7.0f}")

# Summary stats
print("\n" + "=" * 90)
print("FURIA CHECK:")
furia_row = rdf[rdf['team'] == 'FURIA']
if len(furia_row):
    fr = furia_row.iloc[0]
    print(f"  FURIA: #{int(fr['rank'])} (ELO {fr['elo']:.0f}) vs Riot #38")
    print(f"  Previous: #7 (ELO 1657) -- improvement: {1657 - fr['elo']:.0f} ELO points lower")

print("\nLCK teams:")
for _, r in rdf[rdf['league']=='LCK'].iterrows():
    riot_rank = riot.get(r['team'], '?')
    print(f"  #{int(r['rank']):>2} {r['team']:<25} ELO: {r['elo']:.0f}  Riot: #{riot_rank}")

print("\nLPL teams:")
for _, r in rdf[rdf['league']=='LPL'].head(8).iterrows():
    riot_rank = riot.get(r['team'], '?')
    print(f"  #{int(r['rank']):>2} {r['team']:<25} ELO: {r['elo']:.0f}  Riot: #{riot_rank}")
