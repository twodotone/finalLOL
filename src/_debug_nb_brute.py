"""Debug: inspect what features the model sees for Nightbirds vs BRUTE."""
import pandas as pd
import numpy as np
import glob, os, joblib

# Load model and data
model = joblib.load('models/calibrated_xgb.joblib')
df = pd.read_csv('data/processed/model_features_v2.csv')

# Find team IDs
data_dir = 'data/csv'
files = [f for f in glob.glob(os.path.join(data_dir, '*.csv')) if not f.endswith('.bak')]
dfs = [pd.read_csv(f, low_memory=False) for f in files]
raw = pd.concat(dfs, ignore_index=True)

nb = raw[raw['teamname'] == 'Nightbirds']
br = raw[raw['teamname'] == 'BRUTE']
nb_id = nb['teamid'].iloc[-1]
br_id = br['teamid'].iloc[-1]
print(f"Nightbirds ID: {nb_id}")
print(f"BRUTE ID: {br_id}")

nb_feats = df[df['teamid'] == nb_id].iloc[-1]
br_feats = df[df['teamid'] == br_id].iloc[-1]

# Build feature row exactly as cli_predictor does
from features.elo import PlayerEloSystem
# Use the team ELOs from the screenshot
avg_a, avg_b = 1519.1, 1450.0

# Calculate expected from ELO
def expected_score(a, b):
    return 1.0 / (1.0 + 10**((b - a) / 400.0))

p_elo = expected_score(avg_a, avg_b)
print(f"\nPure ELO expected score: {p_elo*100:.1f}%")

stat_names = ['adj_golddiffat15', 'adj_xpdiffat15', 'adj_csdiffat15',
              'adj_firstblood', 'adj_firstdragon', 'adj_firstherald',
              'adj_firsttower', 'adj_firstbaron', 'adj_dpm', 'adj_vspm', 'opp_elo_pre']

row = {'team_elo_pre': avg_a, 'opp_elo_pre': avg_b, 'expected_win_prob': p_elo, 'is_blue_side': 1}
print(f"\n--- Delta Features ---")
for s in stat_names:
    a5 = nb_feats.get(f'roll5_{s}', 0)
    b5 = br_feats.get(f'roll5_{s}', 0)
    a10 = nb_feats.get(f'roll10_{s}', 0)
    b10 = br_feats.get(f'roll10_{s}', 0)
    row[f'delta5_{s}'] = a5 - b5
    row[f'delta10_{s}'] = a10 - b10
    if abs(a5 - b5) > 0.01 or abs(a10 - b10) > 0.01:
        print(f"  {s}: d5={a5-b5:+.2f} (NB:{a5:.2f} BR:{b5:.2f}) | d10={a10-b10:+.2f} (NB:{a10:.2f} BR:{b10:.2f})")

positions = ['top', 'jng', 'mid', 'bot', 'sup']
lane_deltas = []
print(f"\n--- Lane Features ---")
for pos in positions:
    a_lane = nb_feats.get(f'lane_elo_{pos}', 1500)
    b_lane = br_feats.get(f'lane_elo_{pos}', 1500)
    d = a_lane - b_lane
    row[f'lane_delta_{pos}'] = d
    lane_deltas.append(d)
    print(f"  {pos}: NB={a_lane:.1f} BR={b_lane:.1f} delta={d:+.1f}")
row['lane_delta_avg'] = sum(lane_deltas) / len(lane_deltas)
row['lane_delta_worst'] = min(lane_deltas)
print(f"  avg: {row['lane_delta_avg']:+.1f}")
print(f"  worst: {row['lane_delta_worst']:+.1f}")

x_input = pd.DataFrame([row])

# Predict
p_blue = model.predict_proba(x_input)[:, 1][0]
x_input['is_blue_side'] = 0
p_red = model.predict_proba(x_input)[:, 1][0]
p_avg = (p_blue + p_red) / 2.0

print(f"\n--- Model Output ---")
print(f"  P(NB win | blue): {p_blue*100:.1f}%")
print(f"  P(NB win | red):  {p_red*100:.1f}%")
print(f"  P(NB win | avg):  {p_avg*100:.1f}%")
print(f"  Pure ELO:         {p_elo*100:.1f}%")
print(f"  Diff from ELO:    {(p_avg - p_elo)*100:+.1f}pp")

# Feature importances from the model for this prediction
# Let's also check what happens with JUST base features (zero out rolling/lane)
row_elo_only = row.copy()
for k in row_elo_only:
    if k.startswith('delta') or k.startswith('lane'):
        row_elo_only[k] = 0.0
x_elo_only = pd.DataFrame([row_elo_only])
p_elo_only_blue = model.predict_proba(x_elo_only)[:, 1][0]
x_elo_only['is_blue_side'] = 0
p_elo_only_red = model.predict_proba(x_elo_only)[:, 1][0]
p_elo_only = (p_elo_only_blue + p_elo_only_red) / 2.0
print(f"\n  Model (base ELO only, deltas=0): {p_elo_only*100:.1f}%")

# Zero out only lane features
row_no_lane = row.copy()
for k in row_no_lane:
    if k.startswith('lane'):
        row_no_lane[k] = 0.0
x_no_lane = pd.DataFrame([row_no_lane])
p_nl_blue = model.predict_proba(x_no_lane)[:, 1][0]
x_no_lane['is_blue_side'] = 0
p_nl_red = model.predict_proba(x_no_lane)[:, 1][0]
p_no_lane = (p_nl_blue + p_nl_red) / 2.0
print(f"  Model (no lane features):        {p_no_lane*100:.1f}%")
print(f"  Lane feature lift:               {(p_avg - p_no_lane)*100:+.1f}pp")
