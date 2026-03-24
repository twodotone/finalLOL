"""
sweep_predictor.py

Loads the trained sweep XGBoost model and provides a prediction function
that takes the same features computed during match projection.

Returns:
  p_sweep      - probability the series ends in a sweep
  p_competitive - probability the series is competitive (goes to map 3 in BO3)
  verdict      - 'SWEEP WATCH' | 'COMPETITIVE' | 'TOSS UP'
  key_factors  - list of (text, direction) tuples explaining the call
"""
import os
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'sweep_xgb.joblib')

_model_cache = None


def load_sweep_model():
    global _model_cache
    if _model_cache is None and os.path.exists(MODEL_PATH):
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def predict_sweep(
    elo_gap: float,
    win_prob_fav: float,
    delta_gd15_5: float = 0.0,
    delta_dpm_5: float = 0.0,
    delta_vspm_5: float = 0.0,
    delta_fb_5: float = 0.0,
    delta_fd_5: float = 0.0,
    delta_fh_5: float = 0.0,
    delta_ft_5: float = 0.0,
    delta_fbaron_5: float = 0.0,
    delta_gd15_10: float = 0.0,
    delta_dpm_10: float = 0.0,
    avg_dpm_both: float = 0.0,
    fav_lanes_won: float = 2.5,
    fav_lane_avg: float = 0.0,
    fav_lane_worst: float = 0.0,
    series_format: str = 'BO3',
) -> dict:
    """
    Predict series sweep probability given matchup features.

    All delta features are framed as Favorite - Underdog (positive = fav advantage).

    Returns dict with:
      p_sweep, p_competitive, verdict, key_factors, model_used
    """
    model_meta = load_sweep_model()

    key_factors = []

    # Build rule-based signals (always available, model is optional bonus)
    sweep_score = 0  # positive = sweep leaning, negative = competitive

    # ELO gap signal
    if elo_gap >= 200:
        sweep_score += 3
        key_factors.append((f"Large skill gap ({elo_gap:.0f} ELO) — historic lopsided series", 'sweep'))
    elif elo_gap >= 120:
        sweep_score += 1.5
        key_factors.append((f"Clear ELO advantage ({elo_gap:.0f} pts)", 'sweep'))
    elif elo_gap < 60:
        sweep_score -= 2
        key_factors.append((f"Teams closely matched ({elo_gap:.0f} ELO gap)", 'competitive'))

    # Lane count
    if fav_lanes_won >= 4:
        sweep_score += 2
        key_factors.append((f"Favorite dominates {fav_lanes_won:.0f}/5 lanes", 'sweep'))
    elif fav_lanes_won <= 2:
        sweep_score -= 2
        key_factors.append((f"Underdog competitive in lanes ({5 - fav_lanes_won:.0f}/5 lanes)", 'competitive'))

    # Worst lane vulnerability for underdog
    if fav_lane_worst < -100:
        sweep_score -= 1.5
        key_factors.append((f"Favorite has exploitable lane weakness ({fav_lane_worst:.0f} ELO)", 'competitive'))

    # Early game dominance (GD@15)
    if delta_gd15_5 > 1500:
        sweep_score += 2
        key_factors.append((f"Strong early game edge ({delta_gd15_5:+.0f} GD@15)", 'sweep'))
    elif delta_gd15_5 < -300:
        sweep_score -= 1.5
        key_factors.append((f"Underdog competitive early ({delta_gd15_5:+.0f} GD@15)", 'competitive'))

    # High DPM variance = chaotic / coin-flip series
    if avg_dpm_both > 2200:
        sweep_score -= 1.5
        key_factors.append(("High-paced, high-DPM series expected — variance favors underdog", 'competitive'))

    # DPM advantage for fav = snowball potential
    if delta_dpm_5 > 400:
        sweep_score += 1
        key_factors.append((f"Favorite outpaces in damage (+{delta_dpm_5:.0f} DPM)", 'sweep'))

    # If using ML model, override with its calibrated probability
    p_sweep = None
    model_used = False

    if model_meta is not None:
        try:
            features = model_meta['features']
            all_vals = {
                'elo_gap': elo_gap,
                'win_prob_fav': win_prob_fav,
                'delta_gd15_5': delta_gd15_5,
                'delta_dpm_5': delta_dpm_5,
                'delta_vspm_5': delta_vspm_5,
                'delta_fb_5': delta_fb_5,
                'delta_fd_5': delta_fd_5,
                'delta_fh_5': delta_fh_5,
                'delta_ft_5': delta_ft_5,
                'delta_fbaron_5': delta_fbaron_5,
                'delta_gd15_10': delta_gd15_10,
                'delta_dpm_10': delta_dpm_10,
                'avg_dpm_both': avg_dpm_both,
                'fav_lanes_won': fav_lanes_won,
                'fav_lane_avg': fav_lane_avg,
                'fav_lane_worst': fav_lane_worst,
            }
            row = {f: all_vals.get(f, 0.0) for f in features}
            x = pd.DataFrame([row])
            p_sweep = float(model_meta['model'].predict_proba(x)[:, 1][0])
            model_used = True
        except Exception as e:
            print(f"[sweep_predictor] ML prediction failed: {e}")

    # Fallback: convert rule-based score to probability
    if p_sweep is None:
        # Convert sweep_score (typically -5 to +8) to probability
        # sigmoid-like mapping: score=0 → ~50%, score=5 → ~80%, score=-4 → ~30%
        p_sweep = 1 / (1 + np.exp(-sweep_score * 0.35))

    p_competitive = 1 - p_sweep

    # Override key factors if sweep_score diverges strongly from model
    if model_used and p_sweep >= 0.70:
        verdict = '🔵 SWEEP WATCH'
    elif model_used and p_sweep <= 0.38:
        verdict = '🟢 LIKELY COMPETITIVE'
    elif not model_used and sweep_score >= 3:
        verdict = '🔵 SWEEP WATCH'
    elif not model_used and sweep_score <= -2:
        verdict = '🟢 LIKELY COMPETITIVE'
    else:
        verdict = '🟡 COULD GO EITHER WAY'

    # BO5 note
    if series_format == 'BO5':
        key_factors.append(("BO5 format: even dominant teams often drop a game", 'competitive'))
        p_sweep = p_sweep * 0.85  # sweep is less common in BO5
        p_competitive = 1 - p_sweep

    return {
        'p_sweep': round(p_sweep, 3),
        'p_competitive': round(p_competitive, 3),
        'verdict': verdict,
        'key_factors': key_factors,
        'model_used': model_used,
    }
