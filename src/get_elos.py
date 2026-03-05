import os
import pandas as pd
from features.elo import PlayerEloSystem
from cli_predictor import build_elo_engine, find_team

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'csv')
    elo_engine, match_results, players_df = build_elo_engine(data_dir)

    def get_team_elo(team_str):
        name = find_team(team_str, match_results)
        if not name: return 0, name
        team_rows = match_results[match_results['teamname'] == name]
        teamid = team_rows.iloc[-1]['teamid']
        league = team_rows.iloc[-1]['league']
        latest_game = team_rows.iloc[-1]['gameid']
        roster = players_df[players_df['teamid'] == teamid]
        players = roster[roster['gameid'] == latest_game]['playerid'].tolist()
        
        today = pd.Timestamp.now()
        elos = [elo_engine.get_player_elo(pid, today, league) for pid in players]
        avg_elo = sum(elos) / 5.0 if len(elos) == 5 else sum(elos)/len(elos) if len(elos)>0 else 0
        return avg_elo, name

    teams = ['Bilibili Gaming', "Anyone's Legend", 'JD Gaming', 'Weibo Gaming']
    print('\n--- Current LPL ELOs ---')
    for t in teams:
        elo, name = get_team_elo(t)
        print(f'{name or t}: {elo:.1f}')

if __name__ == '__main__':
    main()
