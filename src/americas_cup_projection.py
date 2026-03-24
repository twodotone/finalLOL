"""
Americas Cup Final Projection: Cloud9 vs FURIA
Blends tournament stats into model features for updated prediction.
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli_predictor import build_elo_engine, find_team, bo_win_prob, detect_signals

# ── Americas Cup aggregate stats ──────────────────────────────────────
# Raw per-game averages from the tournament
TOURNEY_STATS = {
    'Cloud9': {
        'games': 8, 'winrate': 0.375,
        'golddiffat15': 948, 'firstblood': 0.50, 'firsttower': 0.625,
        'firstherald': 0.375, 'firstbaron': 0.286, 'dpm': 2535,
        # firstdragon not directly available — approximate from DRA@15
        'firstdragon': 0.50,  # 0.63 drags @15 → ~50% first dragon
    },
    'FURIA': {
        'games': 5, 'winrate': 1.0,
        'golddiffat15': 1088, 'firstblood': 0.60, 'firsttower': 1.0,
        'firstherald': 0.80, 'firstbaron': 0.833, 'dpm': 3040,
        'firstdragon': 0.80,  # 1.20 drags @15 → ~80% first dragon
    },
}
# Stats we can't map from the tournament table — will keep current values:
#   xpdiffat15, csdiffat15, vspm


def sos_multiplier(opp_elo, alpha=0.004):
    """Same sigmoid SOS as the training pipeline."""
    return 2.0 / (1.0 + np.exp(-alpha * (opp_elo - 1500.0)))


def blend_roll5(current_roll5, tourney_raw, tourney_games, opp_elo_at_tourney):
    """
    Blend tournament stats into the roll5 window.
    
    roll5 is a 5-game sliding average of SOS-adjusted stats.
    We replace up to 5 games in the window with tournament data.
    Tournament stats are SOS-adjusted using avg opponent ELO at the event.
    """
    sos = sos_multiplier(opp_elo_at_tourney)
    n_tourney = min(tourney_games, 5)  # can replace at most 5 games in roll5
    n_keep = 5 - n_tourney  # how many old domestic games remain
    weight_new = n_tourney / 5.0
    weight_old = n_keep / 5.0
    
    blended = {}
    mappable = {
        'adj_golddiffat15': 'golddiffat15',
        'adj_firstblood': 'firstblood',
        'adj_firstdragon': 'firstdragon',
        'adj_firstherald': 'firstherald',
        'adj_firsttower': 'firsttower',
        'adj_firstbaron': 'firstbaron',
        'adj_dpm': 'dpm',
    }
    
    for feat_key, raw_key in mappable.items():
        roll5_key = f'roll5_{feat_key}'
        current_val = current_roll5.get(roll5_key, 0)
        tourney_adj = tourney_raw[raw_key] * sos  # SOS-adjust tournament stat
        blended[roll5_key] = weight_old * current_val + weight_new * tourney_adj
    
    # Copy unmappable features as-is (xpdiffat15, csdiffat15, vspm, opp_elo_pre)
    for key in current_roll5:
        if key.startswith('roll5_') and key not in blended:
            blended[key] = current_roll5[key]
    
    return blended


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'csv')
    
    # ── Build ELO engine & load model ──
    elo_engine, bridge, match_results, players_df = build_elo_engine(data_dir)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'calibrated_xgb.joblib')
    ml_model = joblib.load(model_path)
    
    model_df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'model_features_v2.csv')
    model_df = pd.read_csv(model_df_path)
    
    # ── Resolve teams ──
    c9_name = find_team('Cloud9', match_results)
    furia_name = find_team('FURIA', match_results)
    
    def get_team_info(team_name):
        team_rows = match_results[match_results['teamname'] == team_name]
        teamid = team_rows.iloc[-1]['teamid']
        league = team_rows.iloc[-1]['league']
        latest_game = team_rows.iloc[-1]['gameid']
        roster = players_df[players_df['teamid'] == teamid]
        players = roster[roster['gameid'] == latest_game]['playerid'].tolist()
        return teamid, league, players
    
    c9_id, c9_league, c9_players = get_team_info(c9_name)
    fu_id, fu_league, fu_players = get_team_info(furia_name)
    
    today = pd.Timestamp.now()
    c9_elos = {pid: elo_engine.get_player_elo(pid, today, c9_league) for pid in c9_players}
    fu_elos = {pid: elo_engine.get_player_elo(pid, today, fu_league) for pid in fu_players}
    
    avg_c9 = sum(c9_elos.values()) / 5.0
    avg_fu = sum(fu_elos.values()) / 5.0
    
    # Calculate bridge for Americas play (LCS vs CBLOL)
    bridge_adj = bridge.get_bridge_adj(today, 'LCS', 'CBLOL')
    
    # ── Current features (pre-tournament) ──
    c9_feats = model_df[model_df['teamid'] == c9_id].iloc[-1]
    fu_feats = model_df[model_df['teamid'] == fu_id].iloc[-1]
    
    # ── Baseline prediction (no tournament data) ──
    p_base = run_prediction(ml_model, c9_feats, fu_feats, avg_c9, avg_fu, elo_engine, "BASELINE (domestic only)", bridge_adj=bridge_adj)
    
    # ── Blend tournament stats ──
    # Estimate opponent ELO at the Americas Cup for SOS adjustment
    # Americas has NA/BR/LATAM teams — average around 1480-1520
    # C9 played 8 games vs mixed opponents, FURIA played 5
    c9_opp_elo_tourney = 1490   # slightly below 1500, Americas-level opponents
    fu_opp_elo_tourney = 1490
    
    c9_current_roll5 = {col: c9_feats[col] for col in c9_feats.index if col.startswith('roll5_')}
    fu_current_roll5 = {col: fu_feats[col] for col in fu_feats.index if col.startswith('roll5_')}
    
    c9_blended = blend_roll5(c9_current_roll5, TOURNEY_STATS['Cloud9'], 
                              TOURNEY_STATS['Cloud9']['games'], c9_opp_elo_tourney)
    fu_blended = blend_roll5(fu_current_roll5, TOURNEY_STATS['FURIA'], 
                              TOURNEY_STATS['FURIA']['games'], fu_opp_elo_tourney)
    
    # Build synthetic feature rows
    c9_updated = c9_feats.copy()
    fu_updated = fu_feats.copy()
    for k, v in c9_blended.items():
        c9_updated[k] = v
    for k, v in fu_blended.items():
        fu_updated[k] = v
    
    p_updated = run_prediction(ml_model, c9_updated, fu_updated, avg_c9, avg_fu, elo_engine, 
                                "WITH AMERICAS CUP (blended roll5)", bridge_adj=bridge_adj)
    
    # ... [Summary comparison continues] ...
    
    # ── Summary comparison ──
    print("\n" + "=" * 60)
    print(" COMPARISON SUMMARY")
    print("=" * 60)
    
    base_c9_bo5 = bo_win_prob(p_base, 'BO5') * 100
    upd_c9_bo5 = bo_win_prob(p_updated, 'BO5') * 100
    
    print(f"  Baseline    → {c9_name} {p_base*100:.1f}% per-game | {base_c9_bo5:.1f}% BO5")
    print(f"  + Americas  → {c9_name} {p_updated*100:.1f}% per-game | {upd_c9_bo5:.1f}% BO5")
    print(f"  Shift:        {(p_updated - p_base)*100:+.1f} pp per-game | {(upd_c9_bo5 - base_c9_bo5):+.1f} pp BO5")
    print()
    
    # Show key feature changes
    print("  KEY FEATURE CHANGES (roll5, SOS-adjusted):")
    feature_labels = {
        'roll5_adj_golddiffat15': 'GD@15',
        'roll5_adj_firstblood': 'First Blood%',
        'roll5_adj_firstdragon': 'First Dragon%',
        'roll5_adj_firstherald': 'First Herald%',
        'roll5_adj_firsttower': 'First Tower%',
        'roll5_adj_firstbaron': 'First Baron%',
        'roll5_adj_dpm': 'DPM',
    }
    print(f"  {'Feature':<20} {'C9 before':>12} {'C9 after':>12} {'FURIA before':>12} {'FURIA after':>12}")
    for feat_key, label in feature_labels.items():
        c9_old = c9_feats[feat_key]
        c9_new = c9_updated[feat_key]
        fu_old = fu_feats[feat_key]
        fu_new = fu_updated[feat_key]
        print(f"  {label:<20} {c9_old:>12.1f} {c9_new:>12.1f} {fu_old:>12.1f} {fu_new:>12.1f}")
    print("=" * 60)


def run_prediction(ml_model, a_feats, b_feats, avg_a, avg_b, elo_engine, label, bridge_adj=0):
    """Run the XGBoost prediction given feature rows for team A and B."""
    p_elo = elo_engine.calculate_expected_score(avg_a, avg_b)
    
    stat_names = ['adj_golddiffat15', 'adj_xpdiffat15', 'adj_csdiffat15',
                  'adj_firstblood', 'adj_firstdragon', 'adj_firstherald',
                  'adj_firsttower', 'adj_firstbaron', 'adj_dpm', 'adj_vspm', 'opp_elo_pre']
    
    row = {'team_elo_pre': avg_a, 'opp_elo_pre': avg_b, 'expected_win_prob': p_elo, 'is_blue_side': 1}
    # ... delta features ...
    for s in stat_names:
        row[f'delta5_{s}'] = a_feats.get(f'roll5_{s}', 0) - b_feats.get(f'roll5_{s}', 0)
        row[f'delta10_{s}'] = a_feats.get(f'roll10_{s}', 0) - b_feats.get(f'roll10_{s}', 0)
    
    # Lane ELO features calculation (for signals only)
    positions = ['top', 'jng', 'mid', 'bot', 'sup']
    lane_deltas = []
    for pos in positions:
        a_lane = a_feats.get(f'lane_elo_{pos}', 1500)
        b_lane = b_feats.get(f'lane_elo_{pos}', 1500)
        d = a_lane - b_lane
        lane_deltas.append(d)
    
    signal_lane_avg = sum(lane_deltas) / 5
    signal_lane_worst = min(lane_deltas)
    
    x_input = pd.DataFrame([row])
    p_blue = ml_model.predict_proba(x_input)[:, 1][0]
    x_input['is_blue_side'] = 0
    p_red = ml_model.predict_proba(x_input)[:, 1][0]
    p_a = (p_blue + p_red) / 2.0
    
    # Post-ML Bridge adjustment in logit space
    if bridge_adj != 0:
        def to_logit(p): return np.log(p / (1 - p))
        def from_logit(l): return 1 / (1 + np.exp(-l))
        p_a = from_logit(to_logit(np.clip(p_a, 0.001, 0.999)) + bridge_adj * 4.0)
    
    p_bo5 = bo_win_prob(p_a, 'BO5')
    
    print(f"\n{'─'*60}")
    print(f" {label}")
    print(f"{'─'*60}")
    print(f" Cloud9  ELO: {avg_a:.1f}")
    print(f" FURIA   ELO: {avg_b:.1f}")
    print(f" ELO Gap: {avg_a - avg_b:+.1f}")
    print(f" Per-Game:  Cloud9 {p_a*100:.1f}% | FURIA {(1-p_a)*100:.1f}%")
    print(f" BO5:       Cloud9 {p_bo5*100:.1f}% | FURIA {(1-p_bo5)*100:.1f}%")
    
    # Matchup signals
    has_lane = True
    if has_lane:
        signal_lane_deltas_dict = {pos: lane_deltas[i] for i, pos in enumerate(positions)}
        signals = detect_signals(
            signal_lane_deltas_dict, signal_lane_avg, signal_lane_worst,
            avg_a - avg_b, row.get('delta5_adj_golddiffat15', None)
        )
        if signals:
            print(f"\n  SIGNALS:")
            for sig_text, sig_dir in signals:
                team_label = "Cloud9" if sig_dir == 'A' else "FURIA"
                arrow = '>>>' if sig_dir == 'A' else '<<<'
                print(f"  {arrow} [{team_label}] {sig_text}")
    
    return p_a


if __name__ == '__main__':
    main()
