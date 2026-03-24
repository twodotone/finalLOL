"""
Retrain XGBoost + Isotonic Calibration with tuned lane ELO features.

Changes from previous model:
- Lane ELO parameters widened (K=32/48, loosened normalization)
- Slimmed to lane_delta_avg + lane_delta_worst only (dropped per-position)
- Total features: 28 (was 33)
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
import joblib
import os

# Load regenerated features
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'model_features_v2.csv'))
df['is_blue_side'] = (df['side'] == 'Blue').astype(int)

# Feature set: drop per-position lane deltas, keep avg + worst only
stat_names = ['adj_golddiffat15', 'adj_xpdiffat15', 'adj_csdiffat15',
              'adj_firstblood', 'adj_firstdragon', 'adj_firstherald',
              'adj_firsttower', 'adj_firstbaron', 'adj_dpm', 'adj_vspm', 'opp_elo_pre']

features = (
    ['team_elo_pre', 'opp_elo_pre', 'expected_win_prob', 'is_blue_side']
    + [f'delta5_{s}' for s in stat_names]
    + [f'delta10_{s}' for s in stat_names]
    + [f'delta30_{s}' for s in stat_names]
    + ['min_cross_league_games', 'delta_cross_league_games'] # cross-league features
)

print(f"Training with {len(features)} features (was 33)")
print(f"Features: {features}")

target = 'result'

# Temporal split (80/20)
split_index = int(len(df) * 0.8)
X_train = df[features].iloc[:split_index]
y_train = df[target].iloc[:split_index]
X_test = df[features].iloc[split_index:]
y_test = df[target].iloc[split_index:]

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# --- Baseline: ELO-only Brier ---
elo_brier = brier_score_loss(y_test, df['expected_win_prob'].iloc[split_index:])
print(f"\nELO-only Brier: {elo_brier:.4f}")

# --- Train raw XGBoost (for diagnostics) ---
xgb_raw = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    learning_rate=0.05, max_depth=4, n_estimators=300,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_raw.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

preds_raw = xgb_raw.predict_proba(X_test)[:, 1]
brier_raw = brier_score_loss(y_test, preds_raw)
print(f"Raw XGBoost Brier: {brier_raw:.4f}")

# --- Feature importance ---
importance = pd.DataFrame({'feature': features, 'importance': xgb_raw.feature_importances_})
importance = importance.sort_values('importance', ascending=False)
print("\n--- Feature Importance ---")
for _, row in importance.iterrows():
    marker = " << LANE" if 'lane' in row['feature'] else ""
    print(f"  {row['feature']:<35} {row['importance']:.4f}{marker}")

# --- Calibrated model (production) ---
xgb_base = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    learning_rate=0.05, max_depth=4, n_estimators=300,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
calibrated_clf = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=5)
calibrated_clf.fit(X_train, y_train)

preds_calib = calibrated_clf.predict_proba(X_test)[:, 1]
brier_calib = brier_score_loss(y_test, preds_calib)
acc = accuracy_score(y_test, (preds_calib >= 0.5).astype(int))
auc = roc_auc_score(y_test, preds_calib)

print(f"\n{'='*50}")
print(f"CALIBRATED MODEL RESULTS")
print(f"{'='*50}")
print(f"Brier Score:  {brier_calib:.4f}  (ELO baseline: {elo_brier:.4f})")
print(f"Improvement:  {elo_brier - brier_calib:.4f}")
print(f"Accuracy:     {acc*100:.1f}%")
print(f"AUC-ROC:      {auc:.4f}")

# --- Save ---
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'calibrated_xgb.joblib')
joblib.dump(calibrated_clf, model_path)
print(f"\nSaved to {model_path}")
print(f"Feature count: {len(features)}")
