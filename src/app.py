import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Update path so it finds our features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cli_predictor import build_elo_engine, load_ml_model, bo_win_prob

# --- UI Configuration ---
st.set_page_config(page_title="LoL Oracle Engine", layout="wide", page_icon="")

@st.cache_resource(show_spinner=True)
def load_backend():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'csv')
    elo_engine, match_results, players_df = build_elo_engine(data_dir)
    ml_model = load_ml_model()
    
    model_df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'model_features_v2.csv')
    model_df = pd.read_csv(model_df_path) if os.path.exists(model_df_path) else None
    
    # Get alphabetical list of teams
    teams = sorted([t for t in match_results['teamname'].unique() if isinstance(t, str)])
    return elo_engine, match_results, players_df, ml_model, model_df, teams

def get_current_roster(team_name, match_results, players_df):
    team_rows = match_results[match_results['teamname'] == team_name]
    teamid = team_rows.iloc[-1]['teamid']
    league = team_rows.iloc[-1]['league']
    latest_game = team_rows.iloc[-1]['gameid']
    date = team_rows.iloc[-1]['date']
    
    roster = players_df[players_df['teamid'] == teamid]
    players_in_latest = roster[roster['gameid'] == latest_game]['playerid'].tolist()
    return teamid, league, players_in_latest, date

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
    elo_engine, match_results, players_df, ml_model, model_df, TEAMS = load_backend()

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
        
    with col3:
        default_team_b_index = TEAMS.index("Bilibili Gaming") if "Bilibili Gaming" in TEAMS else 1
        team_b = st.selectbox("Red Side Team", TEAMS, index=default_team_b_index)

    if st.button("Calculate Projection", type="primary", use_container_width=True):
        if team_a == team_b:
            st.error("Teams must be different!")
            st.stop()
            
        a_id, a_league, a_players, a_date = get_current_roster(team_a, match_results, players_df)
        b_id, b_league, b_players, b_date = get_current_roster(team_b, match_results, players_df)
        
        today = pd.Timestamp.now()
        a_elos = {pid: elo_engine.get_player_elo(pid, today, a_league) for pid in a_players}
        b_elos = {pid: elo_engine.get_player_elo(pid, today, b_league) for pid in b_players}

        avg_a = sum(a_elos.values()) / 5.0 if len(a_elos) == 5 else sum(a_elos.values())/max(len(a_elos), 1)
        avg_b = sum(b_elos.values()) / 5.0 if len(b_elos) == 5 else sum(b_elos.values())/max(len(b_elos), 1)

        p_a = elo_engine.calculate_expected_score(avg_a, avg_b)
        
        used_ml_flag = False
        if ml_model is not None and model_df is not None:
            try:
                a_feats = model_df[model_df['teamid'] == a_id].iloc[-1]
                b_feats = model_df[model_df['teamid'] == b_id].iloc[-1]
                
                x_blue = pd.DataFrame([{
                    'team_elo_pre': avg_a, 'opp_elo_pre': avg_b, 'expected_win_prob': p_a, 'is_blue_side': 1,
                    # We default to 0 for missing stats on generic lookup, real prod uses full rolling DB
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
                    'roll5_adj_vspm': a_feats.get('roll5_adj_vspm', 0)
                }])
                
                x_red = x_blue.copy()
                x_red['is_blue_side'] = 0
                
                # Predict probability of win
                p_a_ml_blue = ml_model.predict_proba(x_blue)[:, 1][0]
                p_a_ml_red = ml_model.predict_proba(x_red)[:, 1][0]
                
                p_a = (p_a_ml_blue + p_a_ml_red) / 2.0
                used_ml_flag = True
            except Exception as e:
                pass

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
                r_list_a.append({"Player": pname, "Current ELO": round(elo, 1)})
            st.dataframe(pd.DataFrame(r_list_a).set_index("Player"), use_container_width=True)

        # Team B Roster
        with r2:
            st.subheader(f" {team_b} ({avg_b:.1f})")
            st.caption(f"Last Game Recorded: {b_date.date()}")
            r_list_b = []
            for pid, elo in b_elos.items():
                pname = players_df[players_df['playerid'] == pid]['playername'].iloc[-1]
                r_list_b.append({"Player": pname, "Current ELO": round(elo, 1)})
            st.dataframe(pd.DataFrame(r_list_b).set_index("Player"), use_container_width=True)

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
