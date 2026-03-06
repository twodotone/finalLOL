#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering & Point-in-Time Dataset Generation (V3)
# Overhauled pipeline with:
# 1. **Dynamic Regional Gravity** — regional baselines shift after international events
# 2. **Sigmoid SOS Multiplier** — non-linear strength-of-schedule adjustment
# 3. **Symmetric Feature Deltas** — XGBoost sees team A stats *relative to* team B
# 4. **Cross-Regional K-Factor** — international matches learn 2x faster
# 5. **Softened Decay** — 0.1% daily, capped at 50 ELO max

# In[1]:


import pandas as pd
import numpy as np
import glob
import os
import sys
from tqdm import tqdm

sys.path.append('../src')
from features.elo import PlayerEloSystem


# In[11]:


# 1. Load Data
data_dir = '../data/csv/'
files = [f for f in glob.glob(os.path.join(data_dir, '*2024*.csv')) + glob.glob(os.path.join(data_dir, '*2025*.csv')) + glob.glob(os.path.join(data_dir, '*2026*.csv')) if not f.endswith('.bak')]

dfs = [pd.read_csv(f, low_memory=False) for f in files]
df_all = pd.concat(dfs, ignore_index=True)

# Ensure chronological order
df_all['date'] = pd.to_datetime(df_all['date'])
df_all = df_all.sort_values(by='date').reset_index(drop=True)
print(f"Loaded {len(df_all)} total rows (includes 2026 data!).")


# ## 1. ELO Engine (runs first — we need opponent ELO for SOS)

# In[12]:


# 2. Run the ELO Engine FIRST. We need opponent ELO to adjust the raw game stats.
elo_engine = PlayerEloSystem()

# Separate into games and players
match_results = df_all[df_all['position'] == 'team']
players_df = df_all[df_all['position'] != 'team'][['gameid', 'teamid', 'playerid', 'playername']]

grouped_players = players_df.groupby(['gameid', 'teamid'])['playerid'].apply(list).reset_index()
grouped_players_dict = grouped_players.set_index(['gameid', 'teamid'])['playerid'].to_dict()

engine_stats = []

# Track international event deltas for Dynamic Regional Gravity
current_event = None
event_player_elos_pre = {}  # {player_id: elo_before_event}

for name, group in tqdm(match_results.groupby('gameid', sort=False), desc="Processing Matches"):
    if len(group) != 2:
        continue

    date = group['date'].iloc[0]
    league = group['league'].iloc[0]

    team_a = group.iloc[0]
    team_b = group.iloc[1]
    team_a_won = bool(team_a['result'])

    key_a = (name, team_a['teamid'])
    key_b = (name, team_b['teamid'])

    if key_a not in grouped_players_dict or key_b not in grouped_players_dict:
        continue

    players_a = grouped_players_dict[key_a]
    players_b = grouped_players_dict[key_b]

    if len(players_a) != 5 or len(players_b) != 5:
        continue

    # --- Dynamic Regional Gravity: track event transitions ---
    if elo_engine.is_tournament(league):
        if current_event != league:
            # New tournament started — snapshot pre-event ELOs
            if current_event is not None and event_player_elos_pre:
                # Previous event just ended — compute deltas and update baselines
                deltas = {}
                for pid, pre_elo in event_player_elos_pre.items():
                    if pid in elo_engine.players:
                        deltas[pid] = elo_engine.players[pid]['elo'] - pre_elo
                if deltas:
                    elo_engine.recalculate_league_baselines(deltas)
            current_event = league
            event_player_elos_pre = {}
        # Snapshot first appearance in this event
        for pid in players_a + players_b:
            if pid not in event_player_elos_pre and pid in elo_engine.players:
                event_player_elos_pre[pid] = elo_engine.players[pid]['elo']
    else:
        # Domestic match — if we were in a tournament, finalize it
        if current_event is not None and event_player_elos_pre:
            deltas = {}
            for pid, pre_elo in event_player_elos_pre.items():
                if pid in elo_engine.players:
                    deltas[pid] = elo_engine.players[pid]['elo'] - pre_elo
            if deltas:
                elo_engine.recalculate_league_baselines(deltas)
            current_event = None
            event_player_elos_pre = {}

    # Process through Engine
    result_elo = elo_engine.process_match(date, league, players_a, players_b, team_a_won)

    # Snapshot first appearance for new tournament players (post-init)
    if elo_engine.is_tournament(league):
        for pid in players_a + players_b:
            if pid not in event_player_elos_pre:
                event_player_elos_pre[pid] = elo_engine.players[pid]['elo']

    # Store for Team A
    engine_stats.append({
        'gameid': name,
        'teamid': team_a['teamid'],
        'team_elo_pre': result_elo['avg_a_elo_pre'],
        'opp_elo_pre': result_elo['avg_b_elo_pre'],
        'expected_win_prob': result_elo['expected_a'],
    })

    # Store for Team B
    engine_stats.append({
        'gameid': name,
        'teamid': team_b['teamid'],
        'team_elo_pre': result_elo['avg_b_elo_pre'],
        'opp_elo_pre': result_elo['avg_a_elo_pre'],
        'expected_win_prob': result_elo['expected_b'],
    })

# Finalize any remaining open event
if current_event is not None and event_player_elos_pre:
    deltas = {}
    for pid, pre_elo in event_player_elos_pre.items():
        if pid in elo_engine.players:
            deltas[pid] = elo_engine.players[pid]['elo'] - pre_elo
    if deltas:
        elo_engine.recalculate_league_baselines(deltas)

elo_df = pd.DataFrame(engine_stats)
print(f"Processed ELOs for {len(elo_df) // 2} games.")
print(f"Regional baseline shifts: {elo_engine.regional_baseline_shifts}")


# ## 2. Sigmoid SOS + Rolling Stats + Symmetric Deltas

# In[14]:


# 3. Create SOS Adjusted Team Stats with Sigmoid multiplier
team_df = df_all[df_all['position'] == 'team'].copy()

# Base stat columns
stat_cols = [
    'gamelength', 'golddiffat15', 'xpdiffat15', 'csdiffat15',
    'firstblood', 'firstdragon', 'firstherald', 'firsttower', 'firstbaron',
    'dpm', 'vspm'
]

for c in stat_cols:
    team_df[c] = pd.to_numeric(team_df[c], errors='coerce').fillna(0)

# Merge ELO
team_with_elo = pd.merge(team_df, elo_df, on=['gameid', 'teamid'], how='inner')

# --- Sigmoid SOS Multiplier ---
# Centered at 1500, with alpha=0.004 controlling steepness
# sos(1300) ≈ 0.69, sos(1500) = 1.00, sos(1650) ≈ 1.22, sos(1800) ≈ 1.43
SOS_ALPHA = 0.004
team_with_elo['sos_multiplier'] = 2.0 / (1.0 + np.exp(-SOS_ALPHA * (team_with_elo['opp_elo_pre'] - 1500.0)))

# Adjust stats by SOS
adjusted_cols = []
for c in stat_cols:
    if c != 'gamelength':
        adj_col = f'adj_{c}'
        team_with_elo[adj_col] = team_with_elo[c] * team_with_elo['sos_multiplier']
        adjusted_cols.append(adj_col)

adjusted_cols.append('opp_elo_pre')

# Rolling 5-game averages (shifted by 1 to prevent data leak)
def rolling_mean_ignore_leak(x, window):
    return x.shift(1).rolling(window=window, min_periods=1).mean()

rolling5 = team_with_elo.groupby('teamid')[adjusted_cols].transform(rolling_mean_ignore_leak, window=5)
rolling5.columns = [f'roll5_{c}' for c in adjusted_cols]

rolling10 = team_with_elo.groupby('teamid')[adjusted_cols].transform(rolling_mean_ignore_leak, window=10)
rolling10.columns = [f'roll10_{c}' for c in adjusted_cols]

rolling_stats = pd.concat([rolling5, rolling10], axis=1)

# Build per-team feature rows
team_features = pd.concat([
    team_with_elo[['gameid', 'teamid', 'side', 'result', 'team_elo_pre', 'opp_elo_pre', 'expected_win_prob']],
    rolling_stats
], axis=1)

team_features.dropna(inplace=True)

# --- Symmetric Deltas ---
# Self-join each game to pair Team A features with Team B features
# Each gameid has exactly 2 rows. We merge on gameid, cross-joining the pair.
delta_cols = [c for c in rolling_stats.columns]  # includes both roll5_* and roll10_*

# Create opponent lookup: for each (gameid, teamid), find the opponent's row
opp_features = team_features[['gameid', 'teamid'] + delta_cols].copy()
opp_features.columns = ['gameid', 'opp_teamid'] + [f'opp_{c}' for c in delta_cols]

# For each gameid, pair each team with the OTHER team's features
game_pairs = team_features[['gameid', 'teamid']].copy()
opp_map = team_with_elo[['gameid', 'teamid']].drop_duplicates()

# Build opponent teamid mapping per game
def build_opp_map(game_teams):
    """For each (gameid, teamid), find the OTHER teamid in the same game."""
    opp_rows = []
    for gid, grp in game_teams.groupby('gameid'):
        tids = grp['teamid'].tolist()
        if len(tids) == 2:
            opp_rows.append({'gameid': gid, 'teamid': tids[0], 'opp_teamid': tids[1]})
            opp_rows.append({'gameid': gid, 'teamid': tids[1], 'opp_teamid': tids[0]})
    return pd.DataFrame(opp_rows)

opp_mapping = build_opp_map(opp_map)

# Merge to get opponent's rolling stats alongside each team's row
model_df_v2 = pd.merge(team_features, opp_mapping, on=['gameid', 'teamid'], how='inner')
model_df_v2 = pd.merge(model_df_v2, opp_features, on=['gameid', 'opp_teamid'], how='inner')

# Compute deltas: team's rolling stat minus opponent's rolling stat
for col in delta_cols:
    # roll5_adj_golddiffat15 -> delta5_adj_golddiffat15, roll10_adj_X -> delta10_adj_X
    delta_name = col.replace('roll5_', 'delta5_').replace('roll10_', 'delta10_')
    model_df_v2[delta_name] = model_df_v2[col] - model_df_v2[f'opp_{col}']

# Drop raw opponent columns (keep deltas + team's own rolling stats for reference)
opp_cols_to_drop = [f'opp_{c}' for c in delta_cols] + ['opp_teamid']
model_df_v2.drop(columns=opp_cols_to_drop, inplace=True)

model_df_v2.dropna(inplace=True)

# Save
model_df_v2.to_csv('../data/processed/model_features_v2.csv', index=False)

print(f"Final V2 Model Dataset Size: {len(model_df_v2)} team-games.")
print(f"Delta5 columns: {[c for c in model_df_v2.columns if c.startswith('delta5_')]}")
print(f"Delta10 columns: {[c for c in model_df_v2.columns if c.startswith('delta10_')]}")
model_df_v2.head()


# In[13]:


# Let's calculate the current Top 10 Global Teams based on their most recent active rosters
recent_rosters = players_df.sort_values(by='gameid').groupby('teamid').tail(5)

team_elos = []
for teamid, group in recent_rosters.groupby('teamid'):
    if len(group) == 5:
        team_info = match_results[match_results['teamid'] == teamid].iloc[-1]
        teamname = team_info['teamname']
        league = team_info['league']

        elos = [elo_engine.players.get(pid, {}).get('elo', 1500) for pid in group['playerid']]
        current_team_elo = sum(elos) / 5.0

        team_elos.append({
            'Team': teamname,
            'League': league,
            'Current Roster ELO': current_team_elo
        })

team_rankings = pd.DataFrame(team_elos).sort_values(by='Current Roster ELO', ascending=False).reset_index(drop=True)
team_rankings.index += 1
print("--- TOP 10 GLOBAL TEAMS ---")
print(team_rankings.head(10))


# In[15]:


def project_matchup(team_a_name, team_b_name, series_format='BO1'):
    """
    Given two team names, look up their current 5-man roster ELOs
    and calculate win probability, including format adjustment.
    """
    # Find the most recent 5 unique players per team
    def get_current_roster(team_name):
        # Get most recent teamid for this name
        team_rows = match_results[match_results['teamname'].str.lower() == team_name.lower()]
        if team_rows.empty:
            return None, None, None

        teamid = team_rows.iloc[-1]['teamid']
        teamname = team_rows.iloc[-1]['teamname']
        league = team_rows.iloc[-1]['league']

        # Get last 5 player entries for this team
        roster = players_df[players_df['teamid'] == teamid]
        latest_game = roster['gameid'].max()
        players_in_latest = roster[roster['gameid'] == latest_game]['playerid'].tolist()

        return teamid, teamname, league, players_in_latest

    a_id, a_name, a_league, a_players = get_current_roster(team_a_name)
    b_id, b_name, b_league, b_players = get_current_roster(team_b_name)

    if not a_players or not b_players:
        print(f"Could not find one or both teams.")
        return None

    # Get current ELOs for each player
    today = pd.Timestamp.now()
    a_elos = {pid: elo_engine.get_player_elo(pid, today, a_league) for pid in a_players}
    b_elos = {pid: elo_engine.get_player_elo(pid, today, b_league) for pid in b_players}

    avg_a = sum(a_elos.values()) / len(a_elos)
    avg_b = sum(b_elos.values()) / len(b_elos)

    p_a_win_game = elo_engine.calculate_expected_score(avg_a, avg_b)
    p_b_win_game = 1 - p_a_win_game

    # Format adjustment
    def bo_win_prob(p, fmt):
        if fmt == 'BO1': return p
        if fmt == 'BO3': return p**2 * (3 - 2*p)
        if fmt == 'BO5': return p**3 * (6 - 8*p + 3*p**2)
        return p

    p_a_series = bo_win_prob(p_a_win_game, series_format)
    p_b_series = 1 - p_a_series

    elo_diff = avg_a - avg_b

    print(f"\n{'='*55}")
    print(f"  MATCHUP PROJECTION: {a_name} vs {b_name}")
    print(f"  League: {a_league}   |   Format: {series_format}")
    print(f"{'='*55}")
    print(f"\n  {a_name:<28} ELO: {avg_a:.1f}")
    for pid, elo in a_elos.items():
        pname = players_df[players_df['playerid'] == pid]['playername'].iloc[-1] if pid in players_df['playerid'].values else pid
        print(f"    - {pname:<28} {elo:.1f}")

    print(f"\n  {b_name:<28} ELO: {avg_b:.1f}")
    for pid, elo in b_elos.items():
        pname = players_df[players_df['playerid'] == pid]['playername'].iloc[-1] if pid in players_df['playerid'].values else pid
        print(f"    - {pname:<28} {elo:.1f}")

    print(f"\n{'─'*55}")
    print(f"  ELO Differential: {elo_diff:+.1f} in favor of {a_name if elo_diff > 0 else b_name}")
    print(f"\n  Per-Game Win Prob:  {a_name} {p_a_win_game*100:.1f}%  |  {b_name} {p_b_win_game*100:.1f}%")
    print(f"  {series_format} Series Win Prob:  {a_name} {p_a_series*100:.1f}%  |  {b_name} {p_b_series*100:.1f}%")
    print(f"{'='*55}\n")

project_matchup('Bilibili Gaming', 'JD Gaming', series_format='BO3')


# In[16]:


def analyze_polymarket_edge(model_p_win_game, market_lines, series_format='BO3', bankroll_label=""):
    """
    Given model's per-game win probability and a set of Polymarket market prices,
    calculate edge and Kelly Criterion stake for each market.
    """
    p = model_p_win_game
    q = 1 - p

    def bo_win(p, fmt):
        if fmt == 'BO3': return p**2 * (3 - 2*p)
        if fmt == 'BO5': return p**3 * (6 - 8*p + 3*p**2)
        return p

    # Per-game probabilities
    p_team_a_series = bo_win(p, series_format)
    p_team_b_series = 1 - p_team_a_series

    # Handicap: P(team_b wins at least 1 game in BO3) = 1 - P(team_a sweeps 2-0)
    p_a_20 = p ** 2
    p_b_wins_at_least_1_map = 1 - p_a_20

    model_probs = {
        'BLG Series Win': p_team_a_series,
        'JDG Series Win': p_team_b_series,
        'JDG +2.5 Maps (wins at least 1 game)': p_b_wins_at_least_1_map,
        'NO JDG +2.5 Maps (BLG 2-0 sweep)': p_a_20,
    }

    print(f"\n{'='*60}")
    print(f"  POLYMARKET EDGE ANALYSIS — BLG vs JDG ({series_format})")
    print(f"  Model per-game P(BLG win): {p*100:.1f}%")
    print(f"{'='*60}")
    print(f"  {'Market':<42} {'Model':>7} {'Market':>8} {'Edge':>7} {'Half-Kelly':>11}")
    print(f"  {'─'*42} {'─'*7} {'─'*8} {'─'*7} {'─'*11}")

    for market_name, market_price in market_lines.items():
        model_prob = model_probs.get(market_name)
        if model_prob is None:
            continue

        edge = model_prob - market_price

        # Kelly Criterion: f* = (bp - q) / b
        # where b = (1/market_price) - 1 (decimal odds - 1)
        if market_price < 1.0:
            b = (1.0 / market_price) - 1.0
            kelly_full = (b * model_prob - (1 - model_prob)) / b
            kelly_half = max(0, kelly_full / 2)  # Half-Kelly is standard for sports
        else:
            kelly_half = 0.0

        signal = "✅ EDGE" if edge > 0.05 else ("⚠️  SLIGHT" if edge > 0.01 else ("🔴 FADE" if edge < -0.05 else "  PASS"))

        print(f"  {market_name:<42} {model_prob*100:>6.1f}% {market_price*100:>7.1f}% {edge*100:>+6.1f}% {kelly_half*100:>9.1f}%  {signal}")

    print(f"{'='*60}")
    print("  * Half-Kelly = % of bankroll to stake on the bet")
    print("  * Only bet markets where Edge > +5 points\n")

# ─── MARKET INPUTS ───────────────────────────────────────
# BLG match win priced at 73%, JDG +2.5 maps (wins 1+ game) at 84%
market_lines = {
    'BLG Series Win':                           0.73,   # Per poly
    'JDG Series Win':                           0.27,   # Implied from BLG 73%
    'JDG +2.5 Maps (wins at least 1 game)':    0.84,   # Per poly
    'NO JDG +2.5 Maps (BLG 2-0 sweep)':        0.16,   # Implied from above
}

# Model per-game probability is 57.7% for BLG
analyze_polymarket_edge(
    model_p_win_game=0.577,
    market_lines=market_lines,
    series_format='BO3'
)


# In[17]:


def analyze_bo5_markets(p, market_lines):
    """
    Full BO5 probability decomposition.
    p = per-game win probability for Team A (BLG)
    """
    q = 1 - p

    # BO5 exact score probabilities
    # P(3-0): A wins games 1,2,3
    p_30 = p**3

    # P(3-1): A wins 3, loses 1. The loss can be in game 1,2,3 (not game 4, which A wins)
    # = C(3,1) * p^3 * q^1
    p_31 = 3 * (p**3) * q

    # P(3-2): A wins 3, loses 2. Last game is a win.
    # = C(4,2) * p^3 * q^2
    p_32 = 6 * (p**3) * (q**2)

    # Same for B
    p_03 = q**3
    p_13 = 3 * (q**3) * p
    p_23 = 6 * (q**3) * (p**2)

    p_a_series = p_30 + p_31 + p_32
    p_b_series = p_03 + p_13 + p_23

    # JDG +4.5 maps = JDG wins at least 2 games = 1 - P(BLG 3-0)
    p_jdg_wins_2plus = 1 - p_30

    # Verify total = 1
    total = p_30 + p_31 + p_32 + p_03 + p_13 + p_23

    model_probs = {
        'BLG Series Win (BO5)':                   p_a_series,
        'JDG Series Win (BO5)':                   p_b_series,
        'BLG 3-0 Sweep':                          p_30,
        'NO BLG 3-0 Sweep (series goes 4+ games)': p_jdg_wins_2plus,
        'BLG wins in 4 (3-1)':                    p_31,
        'BLG wins in 5 (3-2)':                    p_32,
    }

    print(f"\n{'='*65}")
    print(f"  POLYMARKET EDGE ANALYSIS — BLG vs JDG (BO5)")
    print(f"  Model per-game P(BLG win): {p*100:.1f}%   (Probability sum: {total:.4f})")
    print(f"{'='*65}")
    print(f"  {'Market':<45} {'Model':>7} {'Market':>8} {'Edge':>7} {'Half-Kelly':>10}")
    print(f"  {'─'*45} {'─'*7} {'─'*8} {'─'*7} {'─'*10}")

    for market_name, market_price in market_lines.items():
        model_prob = model_probs.get(market_name)
        if model_prob is None:
            continue

        edge = model_prob - market_price
        b = (1.0 / market_price) - 1.0 if 0 < market_price < 1 else 0
        kelly_full = (b * model_prob - (1 - model_prob)) / b if b > 0 else 0
        kelly_half = max(0, kelly_full / 2)

        signal = "✅ EDGE" if edge > 0.05 else ("⚠️  SLIGHT" if edge > 0.02 else ("🔴 FADE" if edge < -0.05 else "  PASS"))

        print(f"  {market_name:<45} {model_prob*100:>6.1f}% {market_price*100:>7.1f}% {edge*100:>+6.1f}% {kelly_half*100:>8.1f}%  {signal}")

    print(f"{'='*65}")
    print(f"\n  Full BO5 Score Distribution (model):")
    for score, prob in [('BLG 3-0', p_30), ('BLG 3-1', p_31), ('BLG 3-2', p_32),
                         ('JDG 3-0', p_03), ('JDG 3-1', p_13), ('JDG 3-2', p_23)]:
        bar = '█' * int(prob * 40)
        print(f"    {score}:  {prob*100:5.1f}%  {bar}")
    print()

market_lines_bo5 = {
    'BLG Series Win (BO5)':                     0.73,   # Implied from market
    'JDG Series Win (BO5)':                     0.27,
    'BLG 3-0 Sweep':                            0.28,   # Market price given
    'NO BLG 3-0 Sweep (series goes 4+ games)':  0.72,   # Implied
}

analyze_bo5_markets(p=0.577, market_lines=market_lines_bo5)


# In[18]:


# First, let's find the exact team name strings for T1 Academy and Nongshim Academy in the dataset
lckc_teams = match_results[match_results['league'] == 'LCKC']['teamname'].unique()
print(sorted(lckc_teams))


# In[19]:


project_matchup('T1 Esports Academy', 'Nongshim Esports Academy', series_format='BO3')


# In[21]:


def quick_kelly(model_prob, market_price, label):
    edge = model_prob - market_price
    b = (1.0 / market_price) - 1.0  # decimal odds minus 1
    kelly_full = (b * model_prob - (1 - model_prob)) / b
    kelly_half = max(0, kelly_full / 2)
    signal = "✅ BET" if edge > 0.05 else ("⚠️  MARGINAL" if edge > 0.02 else ("🔴 FADE" if edge < -0.05 else "  PASS"))
    print(f"  {label}")
    print(f"    Model: {model_prob*100:.1f}%  |  Market: {market_price*100:.1f}%  |  Edge: {edge*100:+.1f}pts")
    print(f"    Half-Kelly Stake: {kelly_half*100:.1f}% of bankroll   {signal}")
    print()

p_game = 0.632
q_game = 1 - p_game

# BO3 exact probs
p_t1a_series = p_game**2 * (3 - 2*p_game)
p_nsa_series = 1 - p_t1a_series
p_t1a_20     = p_game**2                   # T1A sweeps 2-0
p_nsa_1plus  = 1 - p_t1a_20               # NSA wins at least 1 map

print(f"\n  T1 Esports Academy vs Nongshim Esports Academy — BO3")
print(f"  ─────────────────────────────────────────────────────")
quick_kelly(p_t1a_series, 0.62, "T1A Series Win")
quick_kelly(p_nsa_series, 0.38, "NSA Series Win")
print(f"  --- Map handicap markets (if available) ---")
print(f"  Model P(T1A 2-0 sweep): {p_t1a_20*100:.1f}%")
print(f"  Model P(NSA wins ≥1 map): {p_nsa_1plus*100:.1f}%")
print(f"\n  If market prices NSA +1.5 maps (wins at least 1) below {p_nsa_1plus*100-5:.0f}% → clear edge.")


# In[22]:


def american_to_implied(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def american_to_decimal_b(american_odds):
    """Convert American odds to decimal 'b' for Kelly (profit per $1 staked)."""
    if american_odds > 0:
        return american_odds / 100
    else:
        return 100 / abs(american_odds)

p = 0.632
q = 1 - p

# In a BO5, T1A -1.5 maps means T1A wins by 2+ maps: 3-0 or 3-1
p_30 = p**3
p_31 = 3 * (p**3) * q
p_t1a_minus1_5 = p_30 + p_31   # T1A covers -1.5

american_odds = 115
market_implied = american_to_implied(american_odds)
b = american_to_decimal_b(american_odds)

edge = p_t1a_minus1_5 - market_implied

# Kelly: f* = (b*p - q) / b
kelly_full = (b * p_t1a_minus1_5 - (1 - p_t1a_minus1_5)) / b
kelly_half = max(0, kelly_full / 2)

signal = "✅ BET" if edge > 0.05 else ("⚠️  MARGINAL" if edge > 0.02 else ("🔴 FADE" if edge < -0.05 else "  PASS"))

print(f"\n  T1A -1.5 Maps  |  BO5  |  +{american_odds} American Odds")
print(f"  ──────────────────────────────────────────────────────")
print(f"  Covers if:      T1A wins 3-0 OR 3-1")
print(f"  P(T1A 3-0):     {p_30*100:.1f}%")
print(f"  P(T1A 3-1):     {p_31*100:.1f}%")
print(f"  P(T1A covers):  {p_t1a_minus1_5*100:.1f}%  ← Model")
print(f"  Market implied: {market_implied*100:.1f}%  (from +{american_odds})")
print(f"  Edge:           {edge*100:+.1f} points")
print(f"  Half-Kelly:     {kelly_half*100:.1f}% of bankroll")
print(f"  Signal:         {signal}")

