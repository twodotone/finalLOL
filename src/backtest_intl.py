import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import glob
import os
from sklearn.metrics import brier_score_loss

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

features = [
    'team_elo_pre', 'opp_elo_pre', 'expected_win_prob', 'is_blue_side',
    'roll5_opp_elo_pre', 'roll5_adj_golddiffat15', 'roll5_adj_xpdiffat15', 'roll5_adj_csdiffat15', 
    'roll5_adj_firstblood', 'roll5_adj_firstdragon', 'roll5_adj_firstherald', 
    'roll5_adj_firsttower', 'roll5_adj_firstbaron', 'roll5_adj_dpm', 'roll5_adj_vspm'
]
target = 'result'

# Strict OOS Split
train_mask = df['date'] < '2025-01-01'
intl_leagues = ['WLDs', 'MSI', 'EWC']
test_mask = (df['date'] >= '2025-01-01') & (df['league'].isin(intl_leagues))

df_train = df[train_mask]
df_test = df[test_mask]

X_train, y_train = df_train[features], df_train[target]
X_test, y_test = df_test[features], df_test[target]

print(f'\nTraining OOS model on {len(X_train)} chronological games (Before 2025)...')
print(f'Testing on {len(X_test)} International Games (2025 {", ".join(intl_leagues)})...')

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

# Since international has smaller sample sizes, we'll do 10% bins for cleaner aggregation
bins = np.arange(0, 1.10, 0.10)
labels = [f'{int(b*100):02d}-{int((b+0.10)*100):02d}%' for b in bins[:-1]]
df_test['bin'] = pd.cut(df_test['pred_prob'], bins=bins, labels=labels, right=False)

results = df_test.groupby('bin', observed=False).agg(
    games=('result', 'count'),
    actual_win_rate=('result', 'mean'),
    pred_win_rate=('pred_prob', 'mean')
).fillna(0)

results = results[results['games'] > 0]

print('\n==========================================================')
print(' 2025 OOS BACKTEST: INTERNATIONAL PLAY (WLDs, MSI, EWC) ')
print('==========================================================')
print(f"{'Probability Bin':<15} | {'Games':<8} | {'Forecast Win %':<15} | {'Actual Win %'}")
print('-' * 60)
for idx, row in results.iterrows():
    print(f"{idx:<15} | {int(row['games']):<8} | {row['pred_win_rate']*100:>13.1f}% | {row['actual_win_rate']*100:>11.1f}%")
print('==========================================================')

print(f'Overall OOS Brier Score: {brier_score_loss(df_test["result"], df_test["pred_prob"]):.4f}')
print(f'Baseline ELO Brier:      {brier_score_loss(df_test["result"], df_test["expected_win_prob"]):.4f}')
print('==========================================================\n')