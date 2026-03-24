"""
train_sweep_model.py

Trains an XGBoost classifier to predict series sweep probability.
Target: is_sweep = 1 if the series ended without the winner losing a game.

Uses chronological train/test split for honest backtesting.
Saves: models/sweep_xgb.joblib
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
import joblib
import os

SERIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed', 'series_outcomes.csv')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'sweep_xgb.joblib')

# Feature set for sweep prediction
FEATURES = [
    'elo_gap',            # absolute ELO difference (fav - underdog)
    'win_prob_fav',       # model's pre-series win probability for the favorite
    'delta_gd15_5',       # fav's GD@15 advantage (roll5)
    'delta_dpm_5',        # fav's DPM advantage (roll5)
    'delta_vspm_5',       # fav's vision score advantage
    'delta_fb_5',         # first blood rate advantage
    'delta_fd_5',         # first dragon rate advantage
    'delta_fh_5',         # first herald rate advantage
    'delta_ft_5',         # first tower rate advantage
    'delta_fbaron_5',     # first baron rate advantage
    'delta_gd15_10',      # fav's GD@15 advantage (roll10) — trend stability
    'delta_dpm_10',       # fav's DPM advantage (roll10)
    'avg_dpm_both',       # combined DPM — high = volatile/coinflip series
    'fav_lanes_won',      # how many lanes the fav wins (0-5)
    'fav_lane_avg',       # average lane ELO advantage (fav's perspective)
    'fav_lane_worst',     # worst lane for the fav (negative = exploitable weakness)
]


def load_data():
    df = pd.read_csv(SERIES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    # Keep only series with complete features
    available_features = [f for f in FEATURES if f in df.columns]
    print(f"Feature availability: {len(available_features)}/{len(FEATURES)}")
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  Missing: {missing}")
    df = df.dropna(subset=available_features)
    return df, available_features


def main():
    print("Loading series outcomes...")
    df, features = load_data()
    print(f"Total series with complete features: {len(df)}")
    print(f"Overall sweep rate: {df['is_sweep'].mean()*100:.1f}%")
    print()

    # Chronological 80/20 split
    split_idx = int(len(df) * 0.8)
    X = df[features]
    y = df['is_sweep']
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    date_test = df['date'].iloc[split_idx:]

    print(f"Train: {len(X_train)} series | Test: {len(X_test)} series")
    print(f"  Train sweep rate: {y_train.mean()*100:.1f}%")
    print(f"  Test  sweep rate: {y_test.mean()*100:.1f}%")
    print(f"  Test date range: {date_test.min().date()} to {date_test.max().date()}")
    print()

    # Baseline: predict sweep whenever win_prob_fav > 0.65
    if 'win_prob_fav' in features:
        naive_preds = (df['win_prob_fav'].iloc[split_idx:] > 0.65).astype(float)
        naive_brier = brier_score_loss(y_test, df['win_prob_fav'].iloc[split_idx:])
        naive_acc = accuracy_score(y_test, naive_preds)
        print(f"Naive baseline (predict sweep if fav>65%):")
        print(f"  Accuracy: {naive_acc*100:.1f}%")
        print(f"  Brier (using fav win prob): {naive_brier:.4f}")
        print()

    # --- Raw XGBoost ---
    xgb_raw = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        learning_rate=0.05, max_depth=3, n_estimators=200,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5,  # prevent overfitting on smaller dataset
        random_state=42,
    )
    xgb_raw.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
    preds_raw = xgb_raw.predict_proba(X_test)[:, 1]
    brier_raw = brier_score_loss(y_test, preds_raw)
    acc_raw = accuracy_score(y_test, (preds_raw >= 0.5).astype(int))
    print(f"\nRaw XGBoost — Brier: {brier_raw:.4f} | Acc: {acc_raw*100:.1f}%")

    # --- Feature importance ---
    importance = pd.DataFrame({
        'feature': features,
        'importance': xgb_raw.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n--- Feature Importance ---")
    for _, row in importance.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    # --- Calibrated model ---
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        learning_rate=0.05, max_depth=3, n_estimators=200,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, random_state=42,
    )
    calibrated = CalibratedClassifierCV(estimator=xgb_base, method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)

    preds_calib = calibrated.predict_proba(X_test)[:, 1]
    brier_calib = brier_score_loss(y_test, preds_calib)
    acc_calib = accuracy_score(y_test, (preds_calib >= 0.5).astype(int))
    auc_calib = roc_auc_score(y_test, preds_calib)

    print(f"\n{'='*50}")
    print(f"CALIBRATED SWEEP MODEL RESULTS")
    print(f"{'='*50}")
    print(f"Brier Score:  {brier_calib:.4f}")
    print(f"Accuracy:     {acc_calib*100:.1f}%")
    print(f"AUC-ROC:      {auc_calib:.4f}")

    # --- Calibration by win prob bucket ---
    print(f"\n{'='*50}")
    print("Backtest by favorite win probability bucket:")
    print(f"{'='*50}")
    bucket_df = pd.DataFrame({
        'win_prob_fav': df['win_prob_fav'].iloc[split_idx:].values if 'win_prob_fav' in df.columns else np.zeros(len(y_test)),
        'is_sweep': y_test.values,
        'p_sweep_pred': preds_calib,
    })
    buckets = [0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i+1]
        sub = bucket_df[(bucket_df['win_prob_fav'] >= lo) & (bucket_df['win_prob_fav'] < hi)]
        if len(sub) > 0:
            actual_sweep = sub['is_sweep'].mean()
            pred_sweep = sub['p_sweep_pred'].mean()
            print(f"  {lo*100:.0f}-{hi*100:.0f}% fav:  n={len(sub):3d}  actual sweep={actual_sweep*100:.0f}%  predicted={pred_sweep*100:.0f}%")

    # --- Save ---
    model_meta = {
        'model': calibrated,
        'features': features,
    }
    joblib.dump(model_meta, MODEL_PATH)
    print(f"\nSaved to {MODEL_PATH}")
    print(f"Features used: {features}")


if __name__ == '__main__':
    main()
