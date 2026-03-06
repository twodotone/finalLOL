import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import glob
import os

print('Loading raw data & features...')
data_dir = 'data/csv'
files = [f for f in glob.glob(os.path.join(data_dir, '*.csv')) if not f.endswith('.bak')]
dfs = [pd.read_csv(f, low_memory=False) for f in files]
df_raw = pd.concat(dfs, ignore_index=True)
df_raw['date'] = pd.to_datetime(df_raw['date'])

match_meta = df_raw[df_raw['position'] == 'team'][['gameid', 'teamid', 'date', 'league']].drop_duplicates()

df_feats = pd.read_csv('data/processed/model_features_v2.csv')
df_feats['is_blue_side'] = (df_feats['side'] == 'Blue').astype(int)

df = pd.merge(df_feats, match_meta, on=['gameid', 'teamid'], how='inner')
df = df.sort_values('date')

# Base ELO + side features
base_features = ['team_elo_pre', 'opp_elo_pre', 'expected_win_prob', 'is_blue_side']

# Dual-timescale delta features (interleaved: delta5_stat, delta10_stat per stat)
_stat_names = ['adj_golddiffat15', 'adj_xpdiffat15', 'adj_csdiffat15',
               'adj_firstblood', 'adj_firstdragon', 'adj_firstherald',
               'adj_firsttower', 'adj_firstbaron', 'adj_dpm', 'adj_vspm', 'opp_elo_pre']
delta_features_interleaved = []
for s in _stat_names:
    delta_features_interleaved.append(f'delta5_{s}')
    delta_features_interleaved.append(f'delta10_{s}')

available_cols = set(df_feats.columns)
dual_available = all(c in available_cols for c in delta_features_interleaved)

if dual_available:
    features = base_features + delta_features_interleaved
    print('Using dual-timescale DELTA features (V3.1 pipeline)')
else:
    # Legacy fallback
    features = base_features + [
        'roll5_opp_elo_pre', 'roll5_adj_golddiffat15', 'roll5_adj_xpdiffat15', 'roll5_adj_csdiffat15',
        'roll5_adj_firstblood', 'roll5_adj_firstdragon', 'roll5_adj_firstherald',
        'roll5_adj_firsttower', 'roll5_adj_firstbaron', 'roll5_adj_dpm', 'roll5_adj_vspm'
    ]
    print('Using absolute rolling features (V2 pipeline fallback)')

target = 'result'

# Strict OOS Split
train_mask = df['date'] < '2025-01-01'
leagues = ['LEC', 'LCK', 'LPL', 'LCP']
test_mask = (df['date'] >= '2025-01-01') & (df['date'] < '2026-01-01') & (df['league'].isin(leagues))

df_train = df[train_mask]
df_test = df[test_mask]

X_train, y_train = df_train[features], df_train[target]
X_test, y_test = df_test[features], df_test[target]

print(f'\nTraining OOS model on {len(X_train)} chronological games (Before 2025)...')
print(f'Testing on {len(X_test)} games (2025 LEC, LCK, LPL)...')

xgb_base = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    learning_rate=0.05, max_depth=4, n_estimators=100, 
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
calibrated_clf = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=5)
calibrated_clf.fit(X_train, y_train)

preds = calibrated_clf.predict_proba(X_test)[:, 1]
df_test = df_test.copy()
df_test['pred_prob'] = preds

bins = np.arange(0, 1.05, 0.05)
labels = [f'{int(b*100):02d}-{int((b+0.05)*100):02d}%' for b in bins[:-1]]
df_test['bin'] = pd.cut(df_test['pred_prob'], bins=bins, labels=labels, right=False)

results = df_test.groupby('bin', observed=False).agg(
    games=('result', 'count'),
    actual_win_rate=('result', 'mean'),
    pred_win_rate=('pred_prob', 'mean')
).fillna(0)

results = results[results['games'] > 0]

print('\n==========================================================')
print(' 2025 OOS BACKTEST: LEC, LCK, LPL (Pre-2025 Training) ')
print('==========================================================')
print(f"{'Probability Bin':<15} | {'Games':<8} | {'Forecast Win %':<15} | {'Actual Win %'}")
print('-' * 60)
for idx, row in results.iterrows():
    print(f"{idx:<15} | {int(row['games']):<8} | {row['pred_win_rate']*100:>13.1f}% | {row['actual_win_rate']*100:>11.1f}%")
print('==========================================================')

from sklearn.metrics import brier_score_loss
print(f'Overall OOS Brier Score: {brier_score_loss(df_test["result"], df_test["pred_prob"]):.4f}')
