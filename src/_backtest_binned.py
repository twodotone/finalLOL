"""Per-league binned calibration backtest — Lane ON vs OFF comparison."""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import glob, os

# ── Load data ──────────────────────────────────────────────────
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

base_features = ['team_elo_pre', 'opp_elo_pre', 'expected_win_prob', 'is_blue_side']
_stat_names = ['adj_golddiffat15', 'adj_xpdiffat15', 'adj_csdiffat15',
               'adj_firstblood', 'adj_firstdragon', 'adj_firstherald',
               'adj_firsttower', 'adj_firstbaron', 'adj_dpm', 'adj_vspm', 'opp_elo_pre']
delta_features = []
for s in _stat_names:
    delta_features.append(f'delta5_{s}')
    delta_features.append(f'delta10_{s}')
lane_features = ['lane_delta_top', 'lane_delta_jng', 'lane_delta_mid',
                 'lane_delta_bot', 'lane_delta_sup', 'lane_delta_avg', 'lane_delta_worst']
features_lane = base_features + delta_features + lane_features
features_nolane = base_features + delta_features + lane_features  # same cols, zeroed

train_mask = df['date'] < '2025-01-01'
leagues = ['LEC', 'LCK', 'LPL', 'LCP']
test_mask = (df['date'] >= '2025-01-01') & (df['date'] < '2026-01-01') & (df['league'].isin(leagues))

df_train = df[train_mask].dropna(subset=features_lane + ['result'])
df_test_base = df[test_mask].dropna(subset=features_lane + ['result'])

# ── Train & predict: Lane ON ──────────────────────────────────
X_train_on = df_train[features_lane]
y_train = df_train['result']

xgb_base = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    learning_rate=0.05, max_depth=4, n_estimators=100,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
clf_on = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=5)
clf_on.fit(X_train_on, y_train)

df_on = df_test_base.copy()
df_on['pred_prob'] = clf_on.predict_proba(df_on[features_lane])[:, 1]

# ── Train & predict: Lane OFF (zero lane features) ────────────
df_train_off = df_train.copy()
df_train_off[lane_features] = 0.0
df_test_off = df_test_base.copy()
df_test_off[lane_features] = 0.0

xgb_base2 = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    learning_rate=0.05, max_depth=4, n_estimators=100,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
clf_off = CalibratedClassifierCV(estimator=xgb_base2, method='isotonic', cv=5)
clf_off.fit(df_train_off[features_lane], y_train)

df_off = df_test_off.copy()
df_off['pred_prob'] = clf_off.predict_proba(df_off[features_lane])[:, 1]

# ── Per-league comparison ──────────────────────────────────────
wide_bins = np.arange(0, 1.1, 0.1)
wide_labels = [f'{int(b*100):02d}-{int((b+0.1)*100):02d}%' for b in wide_bins[:-1]]

for league in leagues:
    lg_on = df_on[df_on['league'] == league].copy()
    lg_off = df_off[df_off['league'] == league].copy()
    if len(lg_on) == 0:
        continue
    brier_on = brier_score_loss(lg_on['result'], lg_on['pred_prob'])
    brier_off = brier_score_loss(lg_off['result'], lg_off['pred_prob'])
    brier_diff = brier_on - brier_off

    lg_on['bin'] = pd.cut(lg_on['pred_prob'], bins=wide_bins, labels=wide_labels, right=False)
    lg_off['bin'] = pd.cut(lg_off['pred_prob'], bins=wide_bins, labels=wide_labels, right=False)

    res_on = lg_on.groupby('bin', observed=False).agg(
        games=('result', 'count'), actual=('result', 'mean'), pred=('pred_prob', 'mean')
    ).fillna(0)
    res_off = lg_off.groupby('bin', observed=False).agg(
        games=('result', 'count'), actual=('result', 'mean'), pred=('pred_prob', 'mean')
    ).fillna(0)

    print(f"\n{'='*80}")
    better = '← LANE BETTER' if brier_diff < 0 else '← NO-LANE BETTER' if brier_diff > 0 else ''
    print(f" {league} - {len(lg_on)} games | Lane ON: {brier_on:.4f}  Lane OFF: {brier_off:.4f}  Δ={brier_diff:+.4f} {better}")
    print(f"{'='*80}")
    print(f" {'Bin':<10} | {'Games':>5} |  {'Lane ON':>14}  |  {'Lane OFF':>14}  | {'Δ Diff':>7}")
    print(f" {'':<10} | {'':>5} |  {'Model → Actual':>14}  |  {'Model → Actual':>14}  | {'(ON-OFF)':>7}")
    print(f" {'-'*10}-+-{'-'*5}-+-{'-'*17}-+-{'-'*17}-+-{'-'*7}")

    for lbl in wide_labels:
        r_on = res_on.loc[lbl] if lbl in res_on.index else None
        r_off = res_off.loc[lbl] if lbl in res_off.index else None
        n_on = int(r_on['games']) if r_on is not None else 0
        n_off = int(r_off['games']) if r_off is not None else 0
        if n_on == 0 and n_off == 0:
            continue
        n = max(n_on, n_off)

        if n_on > 0:
            diff_on = r_on['actual'] - r_on['pred']
            on_str = f"{r_on['pred']*100:5.1f}→{r_on['actual']*100:5.1f}"
        else:
            diff_on = 0
            on_str = '     ---     '

        if n_off > 0:
            diff_off = r_off['actual'] - r_off['pred']
            off_str = f"{r_off['pred']*100:5.1f}→{r_off['actual']*100:5.1f}"
        else:
            diff_off = 0
            off_str = '     ---     '

        # Δ diff: positive = lane ON is closer to actual (better calibrated)
        delta = abs(diff_off) - abs(diff_on)
        arrow = '✓' if delta > 0.02 else '✗' if delta < -0.02 else '~'
        print(f" {lbl:<10} | {n:>5} |  {on_str:>14}  |  {off_str:>14}  | {delta*100:>+5.1f}% {arrow}")

# ── Combined summary ───────────────────────────────────────────
brier_all_on = brier_score_loss(df_on['result'], df_on['pred_prob'])
brier_all_off = brier_score_loss(df_off['result'], df_off['pred_prob'])

print(f"\n{'='*80}")
print(f" OVERALL COMPARISON ({len(df_on)} games)")
print(f"{'='*80}")
print(f"  Lane ON  Brier: {brier_all_on:.4f}")
print(f"  Lane OFF Brier: {brier_all_off:.4f}")
print(f"  Difference:     {brier_all_on - brier_all_off:+.4f} {'← LANE BETTER' if brier_all_on < brier_all_off else '← NO-LANE BETTER' if brier_all_on > brier_all_off else ''}")

df_on['bin'] = pd.cut(df_on['pred_prob'], bins=wide_bins, labels=wide_labels, right=False)
df_off['bin'] = pd.cut(df_off['pred_prob'], bins=wide_bins, labels=wide_labels, right=False)

res_on_all = df_on.groupby('bin', observed=False).agg(
    games=('result', 'count'), actual=('result', 'mean'), pred=('pred_prob', 'mean')
).fillna(0)
res_off_all = df_off.groupby('bin', observed=False).agg(
    games=('result', 'count'), actual=('result', 'mean'), pred=('pred_prob', 'mean')
).fillna(0)

print(f"\n {'Bin':<10} | {'Games':>5} |  {'Lane ON':>14}  |  {'Lane OFF':>14}  | {'Δ Diff':>7}")
print(f" {'-'*10}-+-{'-'*5}-+-{'-'*17}-+-{'-'*17}-+-{'-'*7}")
for lbl in wide_labels:
    r_on = res_on_all.loc[lbl]
    r_off = res_off_all.loc[lbl]
    n_on, n_off = int(r_on['games']), int(r_off['games'])
    if n_on == 0 and n_off == 0:
        continue
    n = max(n_on, n_off)
    diff_on = r_on['actual'] - r_on['pred'] if n_on > 0 else 0
    diff_off = r_off['actual'] - r_off['pred'] if n_off > 0 else 0
    on_str = f"{r_on['pred']*100:5.1f}→{r_on['actual']*100:5.1f}" if n_on > 0 else '     ---     '
    off_str = f"{r_off['pred']*100:5.1f}→{r_off['actual']*100:5.1f}" if n_off > 0 else '     ---     '
    delta = abs(diff_off) - abs(diff_on)
    arrow = '✓' if delta > 0.02 else '✗' if delta < -0.02 else '~'
    print(f" {lbl:<10} | {n:>5} |  {on_str:>14}  |  {off_str:>14}  | {delta*100:>+5.1f}% {arrow}")
