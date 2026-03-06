import pandas as pd
import numpy as np
import glob
import os
import sys
import joblib

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from features.elo import PlayerEloSystem

def load_ml_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'calibrated_xgb.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def build_elo_engine(data_dir):
    print("Loading datasets...", end="", flush=True)
    files = [f for f in glob.glob(os.path.join(data_dir, '*2024*.csv')) + 
             glob.glob(os.path.join(data_dir, '*2025*.csv')) + 
             glob.glob(os.path.join(data_dir, '*2026*.csv')) if not f.endswith('.bak')]
    
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values(by='date').reset_index(drop=True)
    print(f" Loaded {len(df_all)} rows.")

    elo_engine = PlayerEloSystem()
    match_results = df_all[df_all['position'] == 'team']
    players_df = df_all[df_all['position'] != 'team'][['gameid', 'teamid', 'playerid', 'playername']]
    
    grouped_players = players_df.groupby(['gameid', 'teamid'])['playerid'].apply(list).reset_index()
    grouped_players_dict = grouped_players.set_index(['gameid', 'teamid'])['playerid'].to_dict()

    print("Processing ELOs for all matches (this takes ~15 seconds)...", flush=True)
    
    for name, group in match_results.groupby('gameid', sort=False):
        if len(group) != 2: continue
        date = group['date'].iloc[0]
        league = group['league'].iloc[0]
        team_a = group.iloc[0]
        team_b = group.iloc[1]
        team_a_won = bool(team_a['result'])
        
        key_a = (name, team_a['teamid'])
        key_b = (name, team_b['teamid'])
        
        if key_a not in grouped_players_dict or key_b not in grouped_players_dict: continue
        players_a = grouped_players_dict[key_a]
        players_b = grouped_players_dict[key_b]
        
        if len(players_a) != 5 or len(players_b) != 5: continue
        elo_engine.process_match(date, league, players_a, players_b, team_a_won)
        
    print("ELO Engine fully updated to today's date!\n")
    return elo_engine, match_results, players_df

def bo_win_prob(p, fmt):
    if fmt == 'BO3': return p**2 * (3 - 2*p)
    if fmt == 'BO5': return p**3 * (10 - 15*p + 6*p**2)
    return p

def find_team(search_str, match_results):
    # exact match case insensitive
    teams = match_results['teamname'].dropna().unique()
    for t in teams:
        if str(t).lower() == search_str.lower():
            return t
            
    # partial match
    for t in teams:
        if search_str.lower() in str(t).lower():
            return t
    return None

def project_matchup(elo_engine, match_results, players_df, model_df, ml_model, team_a_str, team_b_str, series_format='BO3'):
    team_a_name = find_team(team_a_str, match_results)
    team_b_name = find_team(team_b_str, match_results)
    
    if not team_a_name:
        print(f"Could not find team matching: {team_a_str}")
        return
    if not team_b_name:
        print(f"Could not find team matching: {team_b_str}")
        return

    def get_current_roster(team_name):
        team_rows = match_results[match_results['teamname'] == team_name]
        teamid = team_rows.iloc[-1]['teamid']
        league = team_rows.iloc[-1]['league']
        latest_game = team_rows.iloc[-1]['gameid']
        roster = players_df[players_df['teamid'] == teamid]
        players_in_latest = roster[roster['gameid'] == latest_game]['playerid'].tolist()
        return teamid, team_name, league, players_in_latest

    a_id, a_name, a_league, a_players = get_current_roster(team_a_name)
    b_id, b_name, b_league, b_players = get_current_roster(team_b_name)
    
    today = pd.Timestamp.now()
    a_elos = {pid: elo_engine.get_player_elo(pid, today, a_league) for pid in a_players}
    b_elos = {pid: elo_engine.get_player_elo(pid, today, b_league) for pid in b_players}
    
    avg_a = sum(a_elos.values()) / 5.0
    avg_b = sum(b_elos.values()) / 5.0

    p_a = elo_engine.calculate_expected_score(avg_a, avg_b)
    
    # ML Calibration Check
    used_ml_flag = False
    if ml_model is not None and model_df is not None:
        try:
            a_feats = model_df[model_df['teamid'] == a_id].iloc[-1]
            b_feats = model_df[model_df['teamid'] == b_id].iloc[-1]
            
            # Check feature generation — dual timescale (V3.1) vs single (V3) vs legacy (V2)
            has_dual_deltas = 'delta5_adj_golddiffat15' in model_df.columns
            has_single_deltas = 'delta_adj_golddiffat15' in model_df.columns
            
            stat_names = ['adj_golddiffat15', 'adj_xpdiffat15', 'adj_csdiffat15',
                          'adj_firstblood', 'adj_firstdragon', 'adj_firstherald',
                          'adj_firsttower', 'adj_firstbaron', 'adj_dpm', 'adj_vspm', 'opp_elo_pre']
            
            if has_dual_deltas:
                # V3.1: Dual-timescale delta features
                row = {'team_elo_pre': avg_a, 'opp_elo_pre': avg_b, 'expected_win_prob': p_a, 'is_blue_side': 1}
                for s in stat_names:
                    a5 = a_feats.get(f'roll5_{s}', 0)
                    b5 = b_feats.get(f'roll5_{s}', 0)
                    a10 = a_feats.get(f'roll10_{s}', 0)
                    b10 = b_feats.get(f'roll10_{s}', 0)
                    row[f'delta5_{s}'] = a5 - b5
                    row[f'delta10_{s}'] = a10 - b10
                x_input = pd.DataFrame([row])
            elif has_single_deltas:
                # V3: Single-timescale delta features
                row = {'team_elo_pre': avg_a, 'opp_elo_pre': avg_b, 'expected_win_prob': p_a, 'is_blue_side': 1}
                for s in stat_names:
                    row[f'delta_{s}'] = a_feats.get(f'roll5_{s}', 0) - b_feats.get(f'roll5_{s}', 0)
                x_input = pd.DataFrame([row])
            else:
                # V2 fallback: absolute team stats
                x_input = pd.DataFrame([{
                    'team_elo_pre': avg_a, 'opp_elo_pre': avg_b, 'expected_win_prob': p_a, 'is_blue_side': 1,
                    'roll5_opp_elo_pre': a_feats.get('roll5_opp_elo_pre', avg_b),
                    'roll5_adj_golddiffat15': a_feats.get('roll5_adj_golddiffat15', 0),
                    'roll5_adj_xpdiffat15': a_feats.get('roll5_adj_xpdiffat15', 0),
                    'roll5_adj_csdiffat15': a_feats.get('roll5_adj_csdiffat15', 0),
                    'roll5_adj_firstblood': a_feats.get('roll5_adj_firstblood', 0),
                    'roll5_adj_firstdragon': a_feats.get('roll5_adj_firstdragon', 0),
                    'roll5_adj_firstherald': a_feats.get('roll5_adj_firstherald', 0),
                    'roll5_adj_firsttower': a_feats.get('roll5_adj_firsttower', 0),
                    'roll5_adj_firstbaron': a_feats.get('roll5_adj_firstbaron', 0),
                    'roll5_adj_dpm': a_feats.get('roll5_adj_dpm', 0),
                    'roll5_adj_vspm': a_feats.get('roll5_adj_vspm', 0),
                }])
            
            p_a_ml_blue = ml_model.predict_proba(x_input)[:, 1][0]
            
            # Predict Team A win prob assuming Team A is Red side
            x_input['is_blue_side'] = 0
            p_a_ml_red = ml_model.predict_proba(x_input)[:, 1][0]
            
            # Average the sides
            p_a = (p_a_ml_blue + p_a_ml_red) / 2.0
            used_ml_flag = True
        except Exception as e:
            pass
            
    p_a_series = bo_win_prob(p_a, series_format)
    p_b_series = 1 - p_a_series

    print("\n" + "="*60)
    print(f" PROJECTION: {a_name} vs {b_name} ({series_format})")
    if used_ml_flag: print(f" [* XGBoost Calibration Enabled *]")
    print("="*60)
    print(f" {a_name:<25} ELO: {avg_a:.1f}")
    for pid, elo in a_elos.items():
        pname = players_df[players_df['playerid'] == pid]['playername'].iloc[-1]
        print(f"   - {pname:<21} {elo:.1f}")
        
    print(f"\n {b_name:<25} ELO: {avg_b:.1f}")
    for pid, elo in b_elos.items():
        pname = players_df[players_df['playerid'] == pid]['playername'].iloc[-1]
        print(f"   - {pname:<21} {elo:.1f}")
        
    print(f"\n ELO Differential: {abs(avg_a - avg_b):.1f} points")
    print(f" Per-Game Win Prob:  {a_name} {p_a*100:.1f}% | {b_name} {(1-p_a)*100:.1f}%")
    print(f" Series Win Prob :   {a_name} {p_a_series*100:.1f}% | {b_name} {p_b_series*100:.1f}%")
    print("="*60 + "\n")

    return p_a

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'csv')
    elo_engine, match_results, players_df = build_elo_engine(data_dir)
    
    ml_model = load_ml_model()
    model_df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'model_features_v2.csv')
    if os.path.exists(model_df_path):
        model_df = pd.read_csv(model_df_path)
    else:
        model_df = None

    if len(sys.argv) > 2:
        ta = sys.argv[1]
        tb = sys.argv[2]
        fmt = sys.argv[3] if len(sys.argv) > 3 else 'BO3'
        project_matchup(elo_engine, match_results, players_df, model_df, ml_model, ta, tb, fmt)
    else:
        print("Interactive Mode. Type 'exit' to quit.")
        while True:
            ta = input("Team A name: ")
            if ta.lower() == 'exit': break
            tb = input("Team B name: ")
            if tb.lower() == 'exit': break
            fmt = input("Format (BO1, BO3, BO5): ").upper()
            if fmt not in ['BO1', 'BO3', 'BO5']: fmt = 'BO3'

            project_matchup(elo_engine, match_results, players_df, model_df, ml_model, ta, tb, fmt)