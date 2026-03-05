#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering & Point-in-Time Dataset Generation
# In this notebook, we apply two major pipelines to our raw Oracle's Elixir data to prepare it for XGBoost:
# 1. **Contextual Rolling Stats**: Calculate team performance metrics (e.g., Gold diff at 15, First Dragon %, etc.) over their last $N$ games *prior* to the current match (to prevent data leakage).
# 2. **The Global ELO Engine**: Process every match chronologically, pushing players through our Tiered Bipartite ELO system and extracting their pre-match ELO to generate $P(Win_{ELO})$. 
# 
# Finally, we merge these together to form our ultimate training dataset.

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


# ## 1. Rolling Team Stats
# Oracle's Elixir luckily provides a `team` row per team per game that pre-aggregates many useful stats.
# We will compute the rolling average for the 5 games strictly *prior* to the current game.

# In[3]:


print("Moving ELO processing UP in the pipeline so we can use ELO as an SOS multiplier for rolling stats...")


# In[5]:




# In[12]:


# 2. Run the ELO Engine FIRST. We need opponent ELO to adjust the raw game stats.
elo_engine = PlayerEloSystem()

# Separate into games and players
match_results = df_all[df_all['position'] == 'team']
players_df = df_all[df_all['position'] != 'team'][['gameid', 'teamid', 'playerid', 'playername']]

grouped_players = players_df.groupby(['gameid', 'teamid'])['playerid'].apply(list).reset_index()
# Index for faster lookup
grouped_players_dict = grouped_players.set_index(['gameid', 'teamid'])['playerid'].to_dict()

engine_stats = []

# Process all matches (we will use tqdm to show progress)
for name, group in tqdm(match_results.groupby('gameid', sort=False), desc="Processing Matches"):
    if len(group) != 2:
        continue

    # Needs to be sorted natively by groupby but we already sorted df_all upfront
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
        continue # we only want pure 5v5 matches for the ELO system

    # Process through Engine
    result_elo = elo_engine.process_match(date, league, players_a, players_b, team_a_won)

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

elo_df = pd.DataFrame(engine_stats)
print(f"Processed ELOs for {len(elo_df) // 2} games.")


# In[14]:


# 3. Create SOS Adjusted Team Stats
team_df = df_all[df_all['position'] == 'team'].copy()

# Base stat columns
stat_cols = [
    'gamelength', 'golddiffat15', 'xpdiffat15', 'csdiffat15',
    'firstblood', 'firstdragon', 'firstherald', 'firsttower', 'firstbaron',
    'dpm', 'vspm'
]

# Convert strings to numeric
for c in stat_cols:
    team_df[c] = pd.to_numeric(team_df[c], errors='coerce').fillna(0)

# Merge ELO so we know the opponent's ELO for each match
team_with_elo = pd.merge(team_df, elo_df, on=['gameid', 'teamid'], how='inner')

# Create SOS multiplier. We divide by 1500 (average ELO). 
# If opponent ELO is 1650, multiplier is 1.1 (Stats worth 10% more)
# If opponent ELO is 1350, multiplier is 0.9 (Stats worth 10% less)
team_with_elo['sos_multiplier'] = team_with_elo['opp_elo_pre'] / 1500.0

# Adjust stats that should be scaled by opponent strength
# Values like golddiff, xpdiff, csdiff, dpm. 
# For booleans (firstblood etc), scaling makes them "expected value", so 1 kill against strong team = 1.1 FirstBlood points
adjusted_cols = []
for c in stat_cols:
    if c != 'gamelength': # don't SOS adjust time itself
        adj_col = f'adj_{c}'
        team_with_elo[adj_col] = team_with_elo[c] * team_with_elo['sos_multiplier']
        adjusted_cols.append(adj_col)

# We also want to track the rolling average of the opponent's ELO
adjusted_cols.append('opp_elo_pre')

# Calculate rolling averages (shifted by 1 to prevent data leak)
def rolling_mean_ignore_leak(x):
    return x.shift(1).rolling(window=5, min_periods=1).mean()

# Apply grouping
rolling_stats = team_with_elo.groupby('teamid')[adjusted_cols].transform(rolling_mean_ignore_leak)
rolling_stats.columns = [f'roll5_{c}' for c in adjusted_cols]

# Build Final DF
model_df_v2 = pd.concat([
    team_with_elo[['gameid', 'teamid', 'side', 'result', 'team_elo_pre', 'opp_elo_pre', 'expected_win_prob']], 
    rolling_stats
], axis=1)

# Drop initial NA rows
model_df_v2.dropna(inplace=True)

# Save
model_df_v2.to_csv('../data/processed/model_features_v2.csv', index=False)

print(f"Final V2 Model Dataset Size: {len(model_df_v2)} team-games.")
model_df_v2.head()


# In[13]:


# Let's calculate the current Top 10 Global Teams based on their most recent active rosters
recent_rosters = players_df.sort_values(by='gameid').groupby('teamid').tail(5)

team_elos = []
for teamid, group in recent_rosters.groupby('teamid'):
    if len(group) == 5:
        # Get team name and league from match_results
        team_info = match_results[match_results['teamid'] == teamid].iloc[-1]
        teamname = team_info['teamname']
        league = team_info['league']

        # Calculate current aggregate ELO of these 5 players
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

