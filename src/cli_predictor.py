import pandas as pd
import numpy as np
import glob
import os
import sys
import unicodedata
import joblib
import json

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from features.elo import PlayerEloSystem
from features.regional_bridge import RegionalBridge

# ---------------------------------------------------------------------------
# Roster Override Helpers
# ---------------------------------------------------------------------------
_OVERRIDE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'roster_overrides.json'
)

def _load_roster_override(team_name: str) -> dict | None:
    """
    Load roster_overrides.json and return the override for team_name, or None.
    Matches by exact name first, then normalised fuzzy match.
    """
    if not os.path.exists(_OVERRIDE_PATH):
        return None
    try:
        with open(_OVERRIDE_PATH, 'r', encoding='utf-8') as f:
            overrides = json.load(f)
    except Exception:
        return None

    # Exact match
    if team_name in overrides:
        return overrides[team_name]

    # Normalised fuzzy match (strip diacritics, lowercase, alphanum only)
    def _norm(s):
        import unicodedata as _ud, re as _re
        s = _ud.normalize('NFD', s)
        s = ''.join(c for c in s if _ud.category(c) != 'Mn')
        return _re.sub(r'[^a-z0-9]', '', s.lower())

    needle = _norm(team_name)
    for key, val in overrides.items():
        if key == '_meta':
            continue
        if _norm(key) == needle:
            return val

    return None


def _resolve_override_roster(override: dict, team_name: str, league: str,
                             teamid, players_df, elo_engine):
    """
    Convert a roster override dict {pos: player_name} into a pos_map
    {pos: playerid}, looking up each name in players_df.

    - Known players keep their existing ELO (by playerid).
    - Unknown players (true rookies) are auto-initialised at the
      league's baseline ELO the first time they appear in a projection.

    Returns pos_map or None if fewer than 5 positions are resolved.
    """
    import re as _re, unicodedata as _ud

    def _norm(s):
        s = _ud.normalize('NFD', str(s))
        s = ''.join(c for c in s if _ud.category(c) != 'Mn')
        return _re.sub(r'[^a-z0-9]', '', s.lower())

    # Build a normalised name → playerid lookup from the full players_df
    name_to_pid = {}
    for _, row in players_df[['playername', 'playerid']].drop_duplicates().iterrows():
        norm = _norm(row['playername'])
        if norm:
            name_to_pid[norm] = row['playerid']

    positions = ['top', 'jng', 'mid', 'bot', 'sup']
    pos_map = {}
    for pos in positions:
        pname = override.get(pos, '').strip()
        if not pname:
            continue
        pid = name_to_pid.get(_norm(pname))
        if pid is None:
            # Genuine rookie: synthesise a stable playerid from name
            pid = f'__rookie_{_norm(pname)}'
            print(f"  [Override] {pname} ({pos.upper()}) not found in dataset — "
                  f"initialising at {league} baseline (synthetic id: {pid})")
        pos_map[pos] = pid

    if len(pos_map) < 5:
        missing = [p for p in positions if p not in pos_map]
        print(f"  [Override] Warning: could not resolve {missing} for {team_name}. "
              f"Falling back to last-game roster.")
        return None

    return pos_map

def load_ml_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'calibrated_xgb.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def build_elo_engine(data_dir):
    print("Loading datasets...", end="", flush=True)
    files = [f for f in glob.glob(os.path.join(data_dir, '*.csv')) if not f.endswith('.bak')]
    
    dfs = [pd.read_csv(f, low_memory=False) for f in sorted(files)]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values(by='date').reset_index(drop=True)
    print(f" Loaded {len(df_all)} rows.")

    elo_engine = PlayerEloSystem()
    bridge = RegionalBridge(half_life_days=365, max_adj=0.15, min_effective_games=5.0)
    match_results = df_all[df_all['position'] == 'team']
    players_df = df_all[df_all['position'] != 'team'][['gameid', 'teamid', 'playerid', 'playername', 'position']]
    
    grouped_players = players_df.groupby(['gameid', 'teamid'])['playerid'].apply(list).reset_index()
    grouped_players_dict = grouped_players.set_index(['gameid', 'teamid'])['playerid'].to_dict()

    print("Processing ELOs for all matches...", flush=True)
    
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
        result_elo = elo_engine.process_match(date, league, players_a, players_b, team_a_won)

        # Feed cross-regional results to the bridge
        if result_elo['is_cross_region']:
            homes_a = [elo_engine._get_home_league(pid) for pid in players_a]
            homes_b = [elo_engine._get_home_league(pid) for pid in players_b]
            homes_a = [h for h in homes_a if h is not None]
            homes_b = [h for h in homes_b if h is not None]
            region_a = bridge.get_region(Counter(homes_a).most_common(1)[0][0]) if homes_a else None
            region_b = bridge.get_region(Counter(homes_b).most_common(1)[0][0]) if homes_b else None
            if region_a and region_b and region_a != region_b:
                bridge.record_game(date, region_a, region_b,
                                   actual_a=1 if team_a_won else 0,
                                   expected_a=result_elo['expected_a'])
        
    print("ELO Engine + Regional Bridge fully updated!\n")
    return elo_engine, bridge, match_results, players_df

def bo_win_prob(p, fmt):
    if fmt == 'BO3': return p**2 * (3 - 2*p)
    if fmt == 'BO5': return p**3 * (10 - 15*p + 6*p**2)
    return p


def detect_signals(lane_deltas_dict, lane_avg, lane_worst, elo_gap, form_gd5=None):
    """Detect notable patterns from lane ELO and form data.
    Returns list of (signal_str, direction) where direction is 'A' or 'B'.
    """
    signals = []
    positions = ['top', 'jng', 'mid', 'bot', 'sup']
    pos_labels = {'top': 'TOP', 'jng': 'JNG', 'mid': 'MID', 'bot': 'BOT', 'sup': 'SUP'}

    # Historical WR lookup tables (from 41K game analysis)
    lane_threshold_wr = {50: 66, 75: 71, 100: 74, 125: 79, 150: 85}
    n_winning = sum(1 for d in lane_deltas_dict.values() if d > 0)
    n_winning_opp = 5 - n_winning
    lane_count_wr = {0: 25, 1: 35, 2: 45, 3: 55, 4: 65, 5: 75}

    # --- Lane Count Signal ---
    if n_winning >= 4:
        wr = lane_count_wr[n_winning]
        signals.append((f"{n_winning}/5 lanes favored ({wr}% hist WR)", 'A'))
    elif n_winning_opp >= 4:
        wr = lane_count_wr[n_winning_opp]
        signals.append((f"{n_winning_opp}/5 lanes favored ({wr}% hist WR)", 'B'))

    # --- Single-Lane Dominance ---
    for pos in positions:
        d = lane_deltas_dict[pos]
        abs_d = abs(d)
        if abs_d >= 100:
            best_t = max(t for t in lane_threshold_wr if t <= abs_d)
            wr = lane_threshold_wr[best_t]
            direction = 'A' if d > 0 else 'B'
            signals.append((f"{pos_labels[pos]} gap: {'+' if d > 0 else ''}{d:.0f} ELO ({wr}% hist WR at +{best_t})", direction))

    # --- Worst-Lane Vulnerability ---
    if lane_worst < -100:
        worst_pos = min(lane_deltas_dict, key=lane_deltas_dict.get)
        abs_w = abs(lane_worst)
        if abs_w >= 150:
            signals.append((f"VULNERABILITY: {pos_labels[worst_pos]} lane at {lane_worst:.0f} (19% hist WR when worst < -150)", 'A'))
        else:
            signals.append((f"Weak lane: {pos_labels[worst_pos]} at {lane_worst:.0f}", 'A'))
    opp_worst = -max(lane_deltas_dict.values())
    if opp_worst < -100:
        worst_pos_b = max(lane_deltas_dict, key=lane_deltas_dict.get)
        abs_w = abs(opp_worst)
        if abs_w >= 150:
            signals.append((f"VULNERABILITY: {pos_labels[worst_pos_b]} lane at {opp_worst:.0f} (19% hist WR when worst < -150)", 'B'))
        else:
            signals.append((f"Weak lane: {pos_labels[worst_pos_b]} at {opp_worst:.0f}", 'B'))

    # --- Lane vs ELO Divergence ---
    if elo_gap < -50 and lane_avg > 20:
        signals.append((f"LANE UPSET SIGNAL: ELO underdog ({elo_gap:+.0f}) but lane edge (+{lane_avg:.0f}) [+14pt hist lift]", 'A'))
    elif elo_gap > 50 and lane_avg < -20:
        signals.append((f"LANE UPSET SIGNAL: ELO underdog ({-elo_gap:+.0f}) but lane edge (+{abs(lane_avg):.0f}) [+14pt hist lift]", 'B'))

    # --- Form + Lane Convergence ---
    if form_gd5 is not None:
        if form_gd5 > 500 and lane_avg > 25:
            signals.append(("FORM+LANE CONVERGENCE: hot GD@15 trend + lane edge (76% hist WR)", 'A'))
        elif form_gd5 < -500 and lane_avg < -25:
            signals.append(("FORM+LANE CONVERGENCE: hot GD@15 trend + lane edge (76% hist WR)", 'B'))
        elif form_gd5 < -500 and lane_avg > 25:
            signals.append(("FORM DIVERGENCE: cold GD@15 trend but lane advantage (66% hist WR)", 'A'))
        elif form_gd5 > 500 and lane_avg < -25:
            signals.append(("FORM DIVERGENCE: cold GD@15 trend but lane advantage (66% hist WR)", 'B'))

    # --- Strong Overall Lane Edge ---
    if abs(lane_avg) >= 50:
        direction = 'A' if lane_avg > 0 else 'B'
        if abs(lane_avg) >= 75:
            wr = 81 if abs(lane_avg) < 100 else 89
            signals.append((f"DOMINANT LANE EDGE: avg +{abs(lane_avg):.0f} ({wr}% hist WR)", direction))
        else:
            signals.append((f"Strong lane edge: avg +{abs(lane_avg):.0f} (71% hist WR)", direction))

    return signals

def _strip_diacritics(s):
    """Remove diacritics so e.g. 'ą' becomes 'a'."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

import re

def find_team(search_str, match_results):
    # exact match case insensitive
    teams = match_results['teamname'].dropna().unique()
    for t in teams:
        if str(t).lower() == search_str.lower():
            return t
            
    # partial match with word boundaries
    needle = _strip_diacritics(search_str.lower())
    needle_alt = needle.replace(' ', '')
    needle_dots = needle.replace(' ', '.')
    
    pattern = re.compile(rf'(?i)(^|[\s\.\-])({re.escape(needle)}|{re.escape(needle_alt)}|{re.escape(needle_dots)})($|[\s\.\-])')
    
    matches = []
    for t in teams:
        t_clean = _strip_diacritics(str(t).lower())
        if pattern.search(t_clean):
            matches.append(t)
        elif re.sub(r'[^a-z0-9]', '', t_clean) == re.sub(r'[^a-z0-9]', '', needle):
            matches.append(t)
            
    if matches:
        # Return the shortest match (usually the main team vs Academy/Challengers)
        return min(matches, key=len)

    return None

def project_matchup(elo_engine, match_results, players_df, model_df, ml_model, team_a_str, team_b_str, series_format='BO3', use_lane=True, bridge=None):
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
        # If last match was a tournament, find most recent domestic league
        if elo_engine.is_tournament(league):
            for _, row in team_rows.iloc[::-1].iterrows():
                if not elo_engine.is_tournament(row['league']):
                    league = row['league']
                    break

        # --- Roster Override (Leaguepedia) ---
        # Check override file before falling back to last-game CSV roster.
        # This allows new signings to be reflected immediately.
        override = _load_roster_override(team_name)
        if override:
            resolved = _resolve_override_roster(
                override, team_name, league, teamid, players_df, elo_engine
            )
            if resolved:
                print(f"  [Override] Using Leaguepedia roster for {team_name}")
                return teamid, team_name, league, resolved

        roster = players_df[players_df['teamid'] == teamid]
        latest_match_players = roster[roster['gameid'] == latest_game]
        pos_map = latest_match_players.set_index('position')['playerid'].to_dict()
        return teamid, team_name, league, pos_map

    a_id, a_name, a_league, a_players = get_current_roster(team_a_name)
    b_id, b_name, b_league, b_players = get_current_roster(team_b_name)
    
    today = pd.Timestamp.now()
    a_ids = list(a_players.values())
    b_ids = list(b_players.values())
    
    a_elos = {pid: elo_engine.get_player_elo(pid, today, a_league) for pid in a_ids}
    b_elos = {pid: elo_engine.get_player_elo(pid, today, b_league) for pid in b_ids}
    
    avg_a = sum(a_elos.values()) / 5.0 if len(a_elos) == 5 else sum(a_elos.values())/max(len(a_elos), 1)
    avg_b = sum(b_elos.values()) / 5.0 if len(b_elos) == 5 else sum(b_elos.values())/max(len(b_elos), 1)

    p_a = elo_engine.calculate_expected_score(avg_a, avg_b)
    
    # Regional Bridge adjustment for cross-regional matchups
    bridge_adj = 0.0
    if bridge is not None:
        # Resolve region — fall back to players' home leagues if current league is a tournament
        region_a = bridge.get_region(a_league)
        if region_a is None:
            homes = [elo_engine._get_home_league(pid) for pid in a_players]
            homes = [h for h in homes if h is not None]
            if homes:
                region_a = bridge.get_region(Counter(homes).most_common(1)[0][0])
        region_b = bridge.get_region(b_league)
        if region_b is None:
            homes = [elo_engine._get_home_league(pid) for pid in b_players]
            homes = [h for h in homes if h is not None]
            if homes:
                region_b = bridge.get_region(Counter(homes).most_common(1)[0][0])
        bridge_adj = bridge.get_bridge_adj(today, region_a, region_b)
    
    # ML Calibration Check
    used_ml_flag = False
    matchup_signals = []
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
                # Fix: Pass RAW p_a (expected_win_prob) because model was trained on unbridged ELOs
                row = {'team_elo_pre': avg_a, 'opp_elo_pre': avg_b, 'expected_win_prob': p_a, 'is_blue_side': 1}
                for s in stat_names:
                    row[f'delta5_{s}'] = a_feats.get(f'roll5_{s}', 0) - b_feats.get(f'roll5_{s}', 0)
                for s in stat_names:
                    row[f'delta10_{s}'] = a_feats.get(f'roll10_{s}', 0) - b_feats.get(f'roll10_{s}', 0)
                for s in stat_names:
                    row[f'delta30_{s}'] = a_feats.get(f'roll30_{s}', 0) - b_feats.get(f'roll30_{s}', 0)
                # V3.2: Lane ELO matchup features (if available and enabled)
                has_lane = use_lane and ('lane_elo_top' in model_df.columns or 'top' in a_players)
                # Removed adding lane_delta_avg and worst to row
                # V3.3: Cross-league confidence features
                has_confidence = 'min_cross_league_games' in model_df.columns
                if has_confidence:
                    a_intl = a_feats.get('team_intl_games', 0)
                    b_intl = b_feats.get('team_intl_games', 0)
                    row['min_cross_league_games'] = min(a_intl, b_intl)
                    row['delta_cross_league_games'] = a_intl - b_intl
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
            
            # Post-ML Bridge Tether: Apply regional bridge adjustment safely in logit space
            # to avoid probability clamping issues and ensure it's not double-counted
            if bridge_adj != 0:
                def to_logit(p): return np.log(p / (1 - p))
                def from_logit(l): return 1 / (1 + np.exp(-l))
                
                # Clip p_a to avoid log(0)
                p_a_clipped = np.clip(p_a, 0.001, 0.999)
                # bridge_adj is expected in probability space, convert roughly to logit
                # 0.10 adj at 0.50 is ~0.40 logit.
                logit_adj = bridge_adj * 4.0 
                p_a = from_logit(to_logit(p_a_clipped) + logit_adj)

            used_ml_flag = True

            # Gather signal context — use actual Lane ELO features if available
            positions_sig = ['top', 'jng', 'mid', 'bot', 'sup']
            if has_lane and len(a_players) == 5 and len(b_players) == 5:
                signal_lane_deltas = {}
                for pos in positions_sig:
                    signal_lane_deltas[pos] = a_feats.get(f'lane_elo_{pos}', 1500) - b_feats.get(f'lane_elo_{pos}', 1500)
                signal_lane_avg = sum(signal_lane_deltas.values()) / 5
                signal_lane_worst = min(signal_lane_deltas.values())
                signal_elo_gap = avg_a - avg_b
                signal_form_gd5 = row.get('delta5_adj_golddiffat15', None) if has_dual_deltas else None
                matchup_signals = detect_signals(
                    signal_lane_deltas, signal_lane_avg, signal_lane_worst,
                    signal_elo_gap, signal_form_gd5
                )
            else:
                matchup_signals = []
        except Exception as e:
                import sys; print(f"ML prediction error: {e}", file=sys.stderr)
    p_a_series = bo_win_prob(p_a, series_format)
    p_b_series = 1 - p_a_series

    print("\n" + "="*60)
    print(f" PROJECTION: {a_name} vs {b_name} ({series_format})")
    if used_ml_flag: print(f" [* XGBoost Calibration Enabled *]")
    print("="*60)
    # Resolve lane ELOs for display
    a_lane_elos = {}
    b_lane_elos = {}
    if ml_model is not None and model_df is not None and used_ml_flag:
        try:
            positions_display = ['top', 'jng', 'mid', 'bot', 'sup']
            for pos in positions_display:
                pid_a = a_players.get(pos)
                if pid_a:
                    a_lane_elos[pid_a] = a_feats.get(f'lane_elo_{pos}', None)
                pid_b = b_players.get(pos)
                if pid_b:
                    b_lane_elos[pid_b] = b_feats.get(f'lane_elo_{pos}', None)
        except:
            pass

    show_lane = any(v is not None for v in list(a_lane_elos.values()) + list(b_lane_elos.values()))

    print(f" {a_name:<25} ELO: {avg_a:.1f}")
    if show_lane:
        print(f"   {'Name':<21} {'ELO':>6}  {'Lane':>6}")
    for pid, elo in a_elos.items():
        pname = players_df[players_df['playerid'] == pid]['playername'].iloc[-1]
        lane_val = a_lane_elos.get(pid)
        if show_lane and lane_val is not None:
            print(f"   - {pname:<21} {elo:6.1f}  {lane_val:6.1f}")
        else:
            print(f"   - {pname:<21} {elo:.1f}")
        
    print(f"\n {b_name:<25} ELO: {avg_b:.1f}")
    if show_lane:
        print(f"   {'Name':<21} {'ELO':>6}  {'Lane':>6}")
    for pid, elo in b_elos.items():
        pname = players_df[players_df['playerid'] == pid]['playername'].iloc[-1]
        lane_val = b_lane_elos.get(pid)
        if show_lane and lane_val is not None:
            print(f"   - {pname:<21} {elo:6.1f}  {lane_val:6.1f}")
        else:
            print(f"   - {pname:<21} {elo:.1f}")
        
    print(f"\n ELO Differential: {abs(avg_a - avg_b):.1f} points")
    if bridge_adj != 0.0:
        favored = a_name if bridge_adj > 0 else b_name
        print(f" Regional Bridge: {bridge_adj:+.3f} ({favored}'s region overperforms cross-regionally)")
    print(f" Per-Game Win Prob:  {a_name} {p_a*100:.1f}% | {b_name} {(1-p_a)*100:.1f}%")
    print(f" Series Win Prob :   {a_name} {p_a_series*100:.1f}% | {b_name} {p_b_series*100:.1f}%")
    print("="*60)

    # Print signals if available
    if used_ml_flag and matchup_signals:
        print("\n  MATCHUP SIGNALS:")
        print("  " + "-"*50)
        for sig_text, sig_dir in matchup_signals:
            team_label = a_name if sig_dir == 'A' else b_name
            arrow = '>>>' if sig_dir == 'A' else '<<<'
            print(f"  {arrow} [{team_label}] {sig_text}")
        print("  " + "-"*50)
    print()

    return p_a

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'csv')
    elo_engine, bridge, match_results, players_df = build_elo_engine(data_dir)
    
    ml_model = load_ml_model()
    model_df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'model_features_v2.csv')
    if os.path.exists(model_df_path):
        model_df = pd.read_csv(model_df_path)
    else:
        model_df = None

    if len(sys.argv) > 2:
        # Check for --no-lane flag
        args = [a for a in sys.argv[1:] if a != '--no-lane']
        use_lane = '--no-lane' not in sys.argv
        ta = args[0]
        tb = args[1]
        fmt = args[2] if len(args) > 2 else 'BO3'
        if not use_lane:
            print("[Lane ELO disabled]")
        project_matchup(elo_engine, match_results, players_df, model_df, ml_model, ta, tb, fmt, use_lane=use_lane, bridge=bridge)
    else:
        print("Interactive Mode. Type 'exit' to quit.")
        while True:
            ta = input("Team A name: ")
            if ta.lower() == 'exit': break
            tb = input("Team B name: ")
            if tb.lower() == 'exit': break
            fmt = input("Format (BO1, BO3, BO5): ").upper()
            if fmt not in ['BO1', 'BO3', 'BO5']: fmt = 'BO3'

            project_matchup(elo_engine, match_results, players_df, model_df, ml_model, ta, tb, fmt, bridge=bridge)