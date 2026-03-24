import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Update path so it finds our features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cli_predictor import build_elo_engine, load_ml_model, bo_win_prob, detect_signals
from sweep_predictor import predict_sweep, load_sweep_model

# --- UI Configuration ---
st.set_page_config(page_title="LoL Oracle Engine", layout="wide", page_icon="")

@st.cache_resource(show_spinner=True)
def load_backend():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'csv')
    elo_engine, bridge, match_results, players_df = build_elo_engine(data_dir)
    ml_model = load_ml_model()
    
    model_df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'model_features_v2.csv')
    model_df = pd.read_csv(model_df_path) if os.path.exists(model_df_path) else None
    
    # Get alphabetical list of teams
    teams = sorted([t for t in match_results['teamname'].unique() if isinstance(t, str)])
    return elo_engine, bridge, match_results, players_df, ml_model, model_df, teams

def get_current_roster(team_name, match_results, players_df, elo_engine=None):
    team_rows = match_results[match_results['teamname'] == team_name]
    teamid = team_rows.iloc[-1]['teamid']
    league = team_rows.iloc[-1]['league']
    latest_game = team_rows.iloc[-1]['gameid']
    date = team_rows.iloc[-1]['date']
    # If last match was a tournament, find most recent domestic league
    if elo_engine is not None and elo_engine.is_tournament(league):
        for _, row in team_rows.iloc[::-1].iterrows():
            if not elo_engine.is_tournament(row['league']):
                league = row['league']
                break
    
    roster = players_df[players_df['teamid'] == teamid]
    latest_match_players = roster[roster['gameid'] == latest_game]
    pos_map = latest_match_players.set_index('position')['playerid'].to_dict()
    return teamid, league, pos_map, date

@st.cache_data(show_spinner=False)
def generate_power_rankings(_elo_engine, _match_results, _players_df):
    # Only pull teams that have played in the last 180 days (active)
    recent_date = pd.Timestamp.now() - pd.Timedelta(days=180)
    recent_matches = _match_results[_match_results['date'] >= recent_date]
    active_teams = recent_matches['teamname'].dropna().unique()
    
    rankings = []
    today = pd.Timestamp.now()
    
    for t in active_teams:
        try:
            team_rows = _match_results[_match_results['teamname'] == t]
            teamid = team_rows.iloc[-1]['teamid']
            league = team_rows.iloc[-1]['league']
            latest_game = team_rows.iloc[-1]['gameid']
            date = team_rows.iloc[-1]['date']
            # If last match was a tournament, find most recent domestic league
            if _elo_engine.is_tournament(league):
                for _, row in team_rows.iloc[::-1].iterrows():
                    if not _elo_engine.is_tournament(row['league']):
                        league = row['league']
                        break
            
            roster = _players_df[_players_df['teamid'] == teamid]
            players = roster[roster['gameid'] == latest_game]['playerid'].tolist()
            
            # Require at least 4 known players to avoid broken data points
            if len(players) >= 4:
                elos = [_elo_engine.get_player_elo(pid, today, league) for pid in players]
                avg_elo = sum(elos) / len(elos)
                rankings.append({
                    "Team": t,
                    "League": league,
                    "ELO Rating": round(avg_elo, 1),
                    "Last Played": date.date()
                })
        except:
            continue
            
    df = pd.DataFrame(rankings).sort_values(by="ELO Rating", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df

# --- Execution ---
st.title(" LoL Esports Oracle Engine")
st.markdown("XGBoost + Chronological Player ELO Prediction Model (Target Brier: `<0.22`)")

with st.spinner("Loading engine (this takes ~15 seconds on initial boot)..."):
    elo_engine, bridge, match_results, players_df, ml_model, model_df, TEAMS = load_backend()

tab_predict, tab_rank = st.tabs([" Matchup Predictor", " Global Power Rankings"])

with tab_rank:
    st.markdown("###  Global ELO Power Rankings")
    st.caption("Active teams only (played within the last 180 days). Rankings account for multi-year ELO transfer logic, inactivity decay, and regional baseline modifiers.")
    
    with st.spinner("Crunching dynamic point-in-time ELOs..."):
        df_rankings = generate_power_rankings(elo_engine, match_results, players_df)
    
    # Optional League Filter
    all_leagues = sorted(df_rankings['League'].unique().tolist())
    selected_leagues = st.multiselect("Filter by Region/League", all_leagues, default=[])
    
    if selected_leagues:
        filtered_df = df_rankings[df_rankings['League'].isin(selected_leagues)]
    else:
        filtered_df = df_rankings
        
    st.dataframe(
        filtered_df,
        column_config={
            "Team": st.column_config.TextColumn("Team Name"),
            "League": st.column_config.TextColumn("Region"),
            "ELO Rating": st.column_config.NumberColumn("ELO Rating", format="%.1f"),
            "Last Played": st.column_config.DateColumn("Last Match Date")
        },
        use_container_width=True,
        height=600
    )


with tab_predict:
    st.markdown("### Matchup Configuration")
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        # Avoid index error on default
        default_team_a_index = TEAMS.index("Anyone's Legend") if "Anyone's Legend" in TEAMS else 0
        team_a = st.selectbox("Blue Side Team", TEAMS, index=default_team_a_index)
        
    with col2:
        format_sel = st.selectbox("Format", ["BO1", "BO3", "BO5"], index=1)
        use_lane = st.checkbox("Lane ELO", value=True, help="Enable player-lane ELO features")
        
    with col3:
        default_team_b_index = TEAMS.index("Bilibili Gaming") if "Bilibili Gaming" in TEAMS else 1
        team_b = st.selectbox("Red Side Team", TEAMS, index=default_team_b_index)

    if st.button("Calculate Projection", type="primary", use_container_width=True):
        if team_a == team_b:
            st.error("Teams must be different!")
            st.stop()
            
        a_id, a_league, a_players, a_date = get_current_roster(team_a, match_results, players_df, elo_engine)
        b_id, b_league, b_players, b_date = get_current_roster(team_b, match_results, players_df, elo_engine)
        
        today = pd.Timestamp.now()
        a_elos = {pid: elo_engine.get_player_elo(pid, today, a_league) for pid in a_players.values()}
        b_elos = {pid: elo_engine.get_player_elo(pid, today, b_league) for pid in b_players.values()}

        avg_a = sum(a_elos.values()) / 5.0 if len(a_elos) == 5 else sum(a_elos.values())/max(len(a_elos), 1)
        avg_b = sum(b_elos.values()) / 5.0 if len(b_elos) == 5 else sum(b_elos.values())/max(len(b_elos), 1)

        p_a = elo_engine.calculate_expected_score(avg_a, avg_b)
        
        # Regional Bridge adjustment for cross-regional matchups
        bridge_adj = 0.0
        if bridge is not None:
            # Resolve region — fall back to players' home leagues if current league is a tournament
            region_a = bridge.get_region(a_league)
            if region_a is None:
                homes = [elo_engine._get_home_league(pid) for pid in a_players.values()]
                homes = [h for h in homes if h is not None]
                if homes:
                    from collections import Counter
                    region_a = bridge.get_region(Counter(homes).most_common(1)[0][0])
            region_b = bridge.get_region(b_league)
            if region_b is None:
                homes = [elo_engine._get_home_league(pid) for pid in b_players.values()]
                homes = [h for h in homes if h is not None]
                if homes:
                    from collections import Counter
                    region_b = bridge.get_region(Counter(homes).most_common(1)[0][0])
            bridge_adj = bridge.get_bridge_adj(today, region_a, region_b)
        
        used_ml_flag = False
        matchup_signals = []
        a_feats = None
        b_feats = None
        if ml_model is not None and model_df is not None:
            try:
                a_feats = model_df[model_df['teamid'] == a_id].iloc[-1]
                b_feats = model_df[model_df['teamid'] == b_id].iloc[-1]
                
                has_dual_deltas = 'delta5_adj_golddiffat15' in model_df.columns
                has_single_deltas = 'delta_adj_golddiffat15' in model_df.columns
                
                stat_names = ['adj_golddiffat15', 'adj_xpdiffat15', 'adj_csdiffat15',
                              'adj_firstblood', 'adj_firstdragon', 'adj_firstherald',
                              'adj_firsttower', 'adj_firstbaron', 'adj_dpm', 'adj_vspm', 'opp_elo_pre']
                
                if has_dual_deltas:
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
                    # Removed adding lane_delta_avg and worst to row to prevent
                    # XGBoost collinearity with expected_win_prob
                    # V3.3: Cross-league confidence features
                    has_confidence = 'min_cross_league_games' in model_df.columns
                    if has_confidence:
                        a_intl = a_feats.get('team_intl_games', 0)
                        b_intl = b_feats.get('team_intl_games', 0)
                        row['min_cross_league_games'] = min(a_intl, b_intl)
                        row['delta_cross_league_games'] = a_intl - b_intl
                    x_blue = pd.DataFrame([row])
                elif has_single_deltas:
                    row = {'team_elo_pre': avg_a, 'opp_elo_pre': avg_b, 'expected_win_prob': p_a, 'is_blue_side': 1}
                    for s in stat_names:
                        row[f'delta_{s}'] = a_feats.get(f'roll5_{s}', 0) - b_feats.get(f'roll5_{s}', 0)
                    x_blue = pd.DataFrame([row])
                else:
                    x_blue = pd.DataFrame([{
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
                
                x_red = x_blue.copy()
                x_red['is_blue_side'] = 0
                
                p_a_ml_blue = ml_model.predict_proba(x_blue)[:, 1][0]
                p_a_ml_red = ml_model.predict_proba(x_red)[:, 1][0]
                
                p_a = (p_a_ml_blue + p_a_ml_red) / 2.0
                
                # Post-ML Bridge Tether: Apply regional bridge adjustment safely in logit space
                # to avoid probability clamping issues and ensure it's not double-counted
                if bridge_adj != 0:
                    def to_logit(p): return np.log(p / (1 - p))
                    def from_logit(l): return 1 / (1 + np.exp(-l))
                    p_a_clipped = np.clip(p_a, 0.001, 0.999)
                    logit_adj = bridge_adj * 4.0 
                    p_a = from_logit(to_logit(p_a_clipped) + logit_adj)
                    
                used_ml_flag = True

                # Gather signal context — use actual Lane ELO features if available
                matchup_signals = []
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
            except Exception as e:
                import sys; print(f"ML prediction error: {e}", file=sys.stderr)

        p_a_series = bo_win_prob(p_a, format_sel)
        p_b_series = 1 - p_a_series
        
        st.divider()
        
        if used_ml_flag:
            st.success(" XGBoost Roll-5 Model Calibrator Applied Successfully")
        else:
            st.warning(" Using Base ELO (Missing Rolling Stats for Context)")

        # --- Metrics Display ---
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a} Series Win %", f"{p_a_series*100:.1f}%")
        m2.metric("ELO Differential", f"{abs(avg_a - avg_b):.1f} pts", 
                  f"{team_a if avg_a >= avg_b else team_b} favored")
        m3.metric(f"{team_b} Series Win %", f"{p_b_series*100:.1f}%")
        
        st.progress(float(p_a_series))

        st.progress(float(p_a_series))

        # --- Series Competitiveness Section (BO3 / BO5 only) ---
        if format_sel in ('BO3', 'BO5') and used_ml_flag and a_feats is not None and b_feats is not None:
            st.divider()
            st.markdown("### 🗺️ Series Competitiveness Predictor")
            st.caption("Trained on 2,000+ historical series — predicts whether this matchup is likely to be a sweep or go the distance.")

            # Determine who the favorite is from the model's perspective
            fav_is_a = avg_a >= avg_b
            fav_name = team_a if fav_is_a else team_b
            dog_name = team_b if fav_is_a else team_a
            elo_gap = abs(avg_a - avg_b)
            win_prob_fav = p_a if fav_is_a else (1 - p_a)

            # Pull delta features (already in `row` dict from ML section)
            def get_delta(key, fav_is_a_flag):
                val = row.get(key, 0.0) or 0.0
                return val if fav_is_a_flag else -val

            delta_gd15_5  = get_delta('delta5_adj_golddiffat15', fav_is_a)
            delta_dpm_5   = get_delta('delta5_adj_dpm', fav_is_a)
            delta_vspm_5  = get_delta('delta5_adj_vspm', fav_is_a)
            delta_fb_5    = get_delta('delta5_adj_firstblood', fav_is_a)
            delta_fd_5    = get_delta('delta5_adj_firstdragon', fav_is_a)
            delta_fh_5    = get_delta('delta5_adj_firstherald', fav_is_a)
            delta_ft_5    = get_delta('delta5_adj_firsttower', fav_is_a)
            delta_fbaron_5= get_delta('delta5_adj_firstbaron', fav_is_a)
            delta_gd15_10 = get_delta('delta10_adj_golddiffat15', fav_is_a)
            delta_dpm_10  = get_delta('delta10_adj_dpm', fav_is_a)

            # Lane features
            positions_s = ['top', 'jng', 'mid', 'bot', 'sup']
            has_lane_s = 'lane_elo_top' in (a_feats.index if hasattr(a_feats, 'index') else [])
            lane_gaps = []
            fav_lane_avg_val = 0.0
            fav_lane_worst_val = 0.0
            fav_lanes_won_val = 2.5

            if has_lane and len(a_players) == 5 and len(b_players) == 5:
                for pos in positions_s:
                    a_le = a_feats.get(f'lane_elo_{pos}', 1500)
                    b_le = b_feats.get(f'lane_elo_{pos}', 1500)
                    diff = (a_le - b_le) if fav_is_a else (b_le - a_le)
                    lane_gaps.append(diff)
                if lane_gaps:
                    fav_lane_avg_val = sum(lane_gaps) / len(lane_gaps)
                    fav_lane_worst_val = min(lane_gaps)
                    fav_lanes_won_val = sum(1 for g in lane_gaps if g > 0)

            avg_dpm_both = (
                (a_feats.get('roll5_adj_dpm', 0) or 0) +
                (b_feats.get('roll5_adj_dpm', 0) or 0)
            ) / 2

            sweep_result = predict_sweep(
                elo_gap=elo_gap,
                win_prob_fav=win_prob_fav,
                delta_gd15_5=delta_gd15_5,
                delta_dpm_5=delta_dpm_5,
                delta_vspm_5=delta_vspm_5,
                delta_fb_5=delta_fb_5,
                delta_fd_5=delta_fd_5,
                delta_fh_5=delta_fh_5,
                delta_ft_5=delta_ft_5,
                delta_fbaron_5=delta_fbaron_5,
                delta_gd15_10=delta_gd15_10,
                delta_dpm_10=delta_dpm_10,
                avg_dpm_both=avg_dpm_both,
                fav_lanes_won=fav_lanes_won_val,
                fav_lane_avg=fav_lane_avg_val,
                fav_lane_worst=fav_lane_worst_val,
                series_format=format_sel,
            )

            p_sweep = sweep_result['p_sweep']
            p_comp  = sweep_result['p_competitive']
            verdict = sweep_result['verdict']
            factors = sweep_result['key_factors']
            ml_sweep = sweep_result['model_used']

            if ml_sweep:
                st.success("✅ XGBoost Sweep Model Applied")
            else:
                st.warning("⚠️ Sweep model not found — using rule-based signals")

            # Verdict banner
            if 'SWEEP' in verdict:
                st.error(f"**{verdict}** — {fav_name} likely wins convincingly")
            elif 'COMPETITIVE' in verdict:
                st.success(f"**{verdict}** — Series likely goes the distance")
            else:
                st.warning(f"**{verdict}** — Hard to call series length")

            # Sweep / Competitive gauge metrics
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric(f"🧹 Sweep Probability", f"{p_sweep*100:.0f}%",
                       help=f"Probability {fav_name} wins without dropping a game")
            sc2.metric("🤝 Competitive Probability", f"{p_comp*100:.0f}%",
                       help="Probability the series goes to the final decisive map")
            sc3.metric("📊 Favorite Win%", f"{win_prob_fav*100:.1f}%",
                       help="Per-game win probability for the series favorite")

            # Visual progress bar: sweep end ← → competitive end
            st.caption(f"← More likely sweep ({fav_name})   |   More likely competitive →")
            st.progress(float(p_sweep))

            # Key factors
            if factors:
                st.markdown("**Key Factors:**")
                for factor_text, factor_dir in factors:
                    icon = "🔵" if factor_dir == 'sweep' else "🟢"
                    st.markdown(f"- {icon} {factor_text}")

        # --- Debug Features (Hidden by default) ---
        with st.expander("🛠️ Debug Model Features"):
            if used_ml_flag:
                st.write("Raw ML Input Row:")
                st.json(row)
                st.write(f"Bridge Adj: {bridge_adj:+.4f}")
                st.write(f"Raw ELO Win Prob: {row['expected_win_prob']:.4f}")
            else:
                st.write("ML not applied (missing rolling stats).")

        # --- Underdog Alert System ---
        if used_ml_flag and a_feats is not None and b_feats is not None and (p_a_series <= 0.40 or p_b_series <= 0.40):
            st.divider()
            st.markdown("### 🚨 Underdog Alert System")
            
            is_a_underdog = p_a_series <= 0.40
            underdog_name = team_a if is_a_underdog else team_b
            favorite_name = team_b if is_a_underdog else team_a
            
            # Extract features (using roll10 as primary, fallback to roll5)
            gd15 = a_feats.get('roll10_adj_golddiffat15', a_feats.get('roll5_adj_golddiffat15', 0)) - b_feats.get('roll10_adj_golddiffat15', b_feats.get('roll5_adj_golddiffat15', 0))
            dpm = a_feats.get('roll10_adj_dpm', a_feats.get('roll5_adj_dpm', 0)) - b_feats.get('roll10_adj_dpm', b_feats.get('roll5_adj_dpm', 0))
            
            if not is_a_underdog:
                gd15 = -gd15
                dpm = -dpm
                
            worst_lane = -999
            if has_lane and len(a_players) == 5 and len(b_players) == 5:
                positions_sig = ['top', 'jng', 'mid', 'bot', 'sup']
                a_b_lane_deltas = [a_feats.get(f'lane_elo_{pos}', 1500) - b_feats.get(f'lane_elo_{pos}', 1500) for pos in positions_sig]
                if is_a_underdog:
                    worst_lane = min(a_b_lane_deltas)
                else:
                    worst_lane = min([-x for x in a_b_lane_deltas])
            
            # Evaluate signals
            green_flags = 0
            yellow_flags = 0
            reasons = []
            
            if worst_lane > -50:
                green_flags += 1
                reasons.append(f"No glaring weak link (worst lane is only {worst_lane:.0f} ELO behind)")
            else:
                reasons.append(f"Weak link detected (worst lane is {worst_lane:.0f} ELO behind)")
                
            if gd15 > 0:
                green_flags += 1
                reasons.append(f"Superior early game pacing (+{gd15:.0f} GD@15 vs favorite)")
            elif gd15 > -500:
                yellow_flags += 1
                reasons.append(f"Competitive early game ({gd15:.0f} GD@15 diff vs favorite)")
            else:
                reasons.append(f"Weak early game ({gd15:.0f} GD@15 diff vs favorite)")
                
            if dpm > 200:
                green_flags += 1
                reasons.append(f"High variance/bloody playstyle (+{dpm:.0f} DPM vs favorite)")
            elif dpm > 0:
                yellow_flags += 1
                reasons.append(f"Above average pacing (+{dpm:.0f} DPM vs favorite)")
            else:
                reasons.append(f"Plays slower/cleaner than favorite (disadvantageous for underdog)")
                
            # Classify Context
            if green_flags >= 2 or (green_flags == 1 and yellow_flags >= 1):
                st.success(f"🟢 **LIVE DOG:** {underdog_name} has structural advantages that make this a potential trap game for {favorite_name}.")
            elif green_flags == 1 or yellow_flags >= 1:
                st.warning(f"🟡 **WARNING SIGNS:** {underdog_name} is showing some life, but {favorite_name} is still systematically favored.")
            else:
                st.error(f"🔴 **EXPECTED LOSS:** {favorite_name} is winning in all stylistic categories. {underdog_name} expected to lose consistently.")
            
            for r in reasons: 
                st.markdown(f"- {r}")

        # Regional Bridge info for cross-regional matchups
        if bridge_adj != 0.0:
            favored = team_a if bridge_adj > 0 else team_b
            st.info(f"Regional Bridge: **{bridge_adj:+.3f}** — {favored}'s region historically overperforms ELO in cross-regional play")

        # Resolve lane ELOs for roster display
        a_lane_elos = {}
        b_lane_elos = {}
        if used_ml_flag and model_df is not None:
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

        # --- Rosters Display ---
        st.markdown("### Active Rosters (Last Played)")
        r1, r2 = st.columns(2)
        
        # Team A Roster
        with r1:
            st.subheader(f" {team_a} ({avg_a:.1f})")
            st.caption(f"Last Game Recorded: {a_date.date()}")
            r_list_a = []
            for pid, elo in a_elos.items():
                pname = players_df[players_df['playerid'] == pid]['playername'].iloc[-1]
                entry = {"Player": pname, "Player ELO": round(elo, 1)}
                lane_val = a_lane_elos.get(pid)
                if show_lane:
                    entry["Lane ELO"] = round(lane_val, 1) if lane_val is not None else "-"
                r_list_a.append(entry)
            if r_list_a:
                st.dataframe(pd.DataFrame(r_list_a).set_index("Player"), use_container_width=True)
            else:
                st.warning("No roster data found.")

        # Team B Roster
        with r2:
            st.subheader(f" {team_b} ({avg_b:.1f})")
            st.caption(f"Last Game Recorded: {b_date.date()}")
            r_list_b = []
            for pid, elo in b_elos.items():
                pname = players_df[players_df['playerid'] == pid]['playername'].iloc[-1]
                entry = {"Player": pname, "Player ELO": round(elo, 1)}
                lane_val = b_lane_elos.get(pid)
                if show_lane:
                    entry["Lane ELO"] = round(lane_val, 1) if lane_val is not None else "-"
                r_list_b.append(entry)
            if r_list_b:
                st.dataframe(pd.DataFrame(r_list_b).set_index("Player"), use_container_width=True)
            else:
                st.warning("No roster data found.")

        # --- Matchup Signals Display ---
        if used_ml_flag and matchup_signals:
            st.divider()
            st.markdown("### Matchup Signals")
            for sig_text, sig_dir in matchup_signals:
                team_label = team_a if sig_dir == 'A' else team_b
                icon = ">>>" if sig_dir == 'A' else "<<<"
                if any(kw in sig_text.upper() for kw in ['DOMINANT', 'CONVERGENCE', 'UPSET', 'VULNERABILITY']):
                    st.warning(f"{icon} **[{team_label}]** {sig_text}")
                else:
                    st.info(f"{icon} **[{team_label}]** {sig_text}")

        # EV Calculator feature
        st.divider()
        st.markdown("### Expected Value (EV) Calculator")
        ev1, ev2 = st.columns(2)
        with ev1:
            st.markdown(f"**Betting on: {team_a}**")
            odds_book = st.number_input("Sportsbook Odds (e.g., +150 or -110) for Team A:", value=100, step=10)
            
            # Convert American odds to Implied Probability
            if odds_book > 0:
                implied_prob = 100 / (odds_book + 100)
            else:
                implied_prob = abs(odds_book) / (abs(odds_book) + 100)
                
            edge = p_a_series - implied_prob
            st.metric("Model Edge", f"{edge*100:+.1f}%", f"Implied Book Prob: {implied_prob*100:.1f}%")
            if edge > 0.05:
                st.success(" High Value +EV Bet")
            elif edge > 0:
                st.info(" Slight +EV Bet")
            else:
                st.error(" Negative EV (Fade)")
