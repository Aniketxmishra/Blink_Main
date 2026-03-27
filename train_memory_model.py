"""
train_memory_model.py  —  Activation-Aware Memory Model Training

Key fix (Priority 1):
  activation_memory_mb had 0.94 correlation with peak_memory_mb but was NEVER
  in the feature set. The old model only saw parameter/weight features and learned
  a near-flat mapping. Adding activation features gives the model the information
  it needs to predict memory correctly across batch sizes and architectures.

New features added:
  - activation_memory_mb        : measured GPU activation memory (corr=0.94)
  - weight_memory_mb            : model weights resident on GPU
  - activation_memory_per_sample: activation_memory_mb / batch_size (batch-normalised)
  - flops_per_activation_mb     : compute intensity relative to activation cost
  - input_resolution_factor     : H*W from input_shape (spatial dimension proxy)
"""
import ast
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Feature columns ──────────────────────────────────────────────────────────
# Ordered by expected importance (helps debug feature importance plots)
FEATURE_COLS = [
    # Activation features (NEW — highest predictive power)
    'activation_memory_mb',          # corr=0.94 with peak_memory_mb
    'weight_memory_mb',              # corr=0.63
    'activation_memory_per_sample',  # activation_mb / batch_size (normalised)
    'flops_per_activation_mb',       # flops / activation_memory_mb

    # Batch & compute
    'batch_size',
    'flops',
    'compute_memory_ratio',

    # Architecture
    'model_size_mb',
    'num_conv_layers',
    'num_fc_layers',
    'num_bn_layers',
    'avg_conv_kernel_size',
    'max_conv_channels',
    'total_conv_params',
    'total_fc_params',
    'model_depth',

    # Spatial
    'input_resolution_factor',       # H*W derived from input_shape string
]


def _parse_input_resolution(shape_str):
    """Parse input_shape string like '(3, 224, 224)' -> H*W = 50176."""
    try:
        shape = ast.literal_eval(str(shape_str))
        if len(shape) >= 3:
            return int(shape[-2]) * int(shape[-1])  # H * W
        return 224 * 224  # default
    except Exception:
        return 224 * 224


def load_and_engineer_features(data_path='data/enriched/enriched_data.csv'):
    """Load enriched data and engineer activation-aware features."""
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found.")
        return None

    df = pd.read_csv(data_path)

    required = ['peak_memory_mb']
    for col in required:
        if col not in df.columns:
            print(f"ERROR: '{col}' not found. Run collect_data.py with GPU enabled.")
            return None

    # ── Filter bad rows ───────────────────────────────────────────────────────
    df = df[df['peak_memory_mb'] > 0].copy()
    if 'timing_cv' in df.columns:
        n_before = len(df)
        df = df[df['timing_cv'] <= 0.15]
        print(f"Dropped {n_before - len(df)} rows with timing_cv > 0.15")

    print(f"Dataset: {len(df)} rows, {df['model_name'].nunique()} architectures")
    print(f"  Architectures: {sorted(df['model_name'].unique())}")

    # ── Engineer new features ─────────────────────────────────────────────────

    # activation_memory_per_sample: removes the batch-size confound
    if 'activation_memory_mb' in df.columns:
        df['activation_memory_per_sample'] = (
            df['activation_memory_mb'] / df['batch_size'].clip(lower=1)
        )
    else:
        # Fallback: estimate from flops (activation mem ≈ 2 bytes × flops / (ops per byte))
        df['activation_memory_mb'] = df['flops'] / (1024**2) * 4 / 1e3
        df['activation_memory_per_sample'] = df['activation_memory_mb'] / df['batch_size']
        print("  WARNING: activation_memory_mb not found, using FLOPs-based estimate")

    if 'weight_memory_mb' not in df.columns:
        df['weight_memory_mb'] = df['model_size_mb']  # param memory ≈ model size

    # flops per MB of activation (compute intensity relative to activation cost)
    df['flops_per_activation_mb'] = (
        df['flops'] / df['activation_memory_mb'].clip(lower=1e-3)
    )

    # Spatial resolution feature from input_shape string
    if 'input_shape' in df.columns:
        df['input_resolution_factor'] = df['input_shape'].apply(_parse_input_resolution)
    else:
        df['input_resolution_factor'] = 224 * 224

    return df


def train_memory_model(df, target='peak_memory_mb', test_size=0.2, random_state=42):
    """Train activation-aware XGBoost memory model with Optuna tuning."""
    # Use only available feature columns
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Note: {len(missing)} features not available: {missing}")

    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols].fillna(0)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        # Stratify by model_name if available to avoid architecture leakage
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # ── Optuna hyperparameter search ──────────────────────────────────────────
    print("\nTuning XGBoost via Optuna (40 trials)...")

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 50, 400),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            'random_state': random_state,
        }
        Xtr, Xval, ytr, yval = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state
        )
        m = xgb.XGBRegressor(**params, verbosity=0)
        m.fit(Xtr, ytr)
        pred = m.predict(Xval)
        return float(np.sqrt(mean_squared_error(yval, pred)))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=40)
    best = study.best_params
    best['random_state'] = random_state
    print(f"Best params: {best}")

    # ── Train final models ────────────────────────────────────────────────────
    model = xgb.XGBRegressor(**best, verbosity=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print("Memory Model Performance (Activation-Aware v2):")
    print(f"  RMSE : {rmse:.2f} MB")
    print(f"  MAE  : {mae:.2f} MB")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*50}")

    # Quantile models for 80% confidence interval (10th / 90th percentile)
    print("\nTraining quantile models for uncertainty bounds...")
    model_lower = xgb.XGBRegressor(
        **best, objective='reg:quantileerror', quantile_alpha=0.1, verbosity=0
    )
    model_lower.fit(X_train, y_train)

    model_upper = xgb.XGBRegressor(
        **best, objective='reg:quantileerror', quantile_alpha=0.9, verbosity=0
    )
    model_upper.fit(X_train, y_train)

    return model, {'lower': model_lower, 'upper': model_upper}, \
           X_test, y_test, feature_cols


def visualize_results(model, X_test, y_test, feature_cols):
    """Generate diagnostics plots."""
    os.makedirs('results', exist_ok=True)
    y_pred = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', linewidth=0.5)
    lo = min(y_test.min(), y_pred.min()); hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Peak Memory (MB)')
    ax.set_ylabel('Predicted Peak Memory (MB)')
    ax.set_title('Memory Model v2 — Actual vs Predicted')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Feature Importance
    ax = axes[1]
    importances = model.feature_importances_
    idx = np.argsort(importances)
    ax.barh(range(len(idx)), importances[idx])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_cols[i] for i in idx], fontsize=8)
    ax.set_xlabel('Feature Importance (gain)')
    ax.set_title('Memory Model v2 — Feature Importance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/memory_v2_diagnostics.png', dpi=150)
    plt.close()
    print("  Saved: results/memory_v2_diagnostics.png")


def save_feature_list(feature_cols, path='models/memory_model_features.json'):
    """Save the exact feature list so the predictor knows what to build at inference."""
    import json
    os.makedirs('models', exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'features': feature_cols, 'version': 'v2_activation_aware'}, f, indent=2)
    print(f"  Feature list saved: {path}")


def main():
    print("=" * 60)
    print("Blink Memory Model — Priority 1 Retrain")
    print("New features: activation_memory_mb, weight_memory_mb,")
    print("              activation_memory_per_sample, flops_per_activation_mb")
    print("=" * 60)

    print("\nLoading & engineering features...")
    df = load_and_engineer_features()
    if df is None or len(df) == 0:
        print("Failed to load data. Exiting.")
        return

    model, bound_models, X_test, y_test, feature_cols = train_memory_model(df)
    visualize_results(model, X_test, y_test, feature_cols)

    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/memory_model.joblib')
    joblib.dump(bound_models['lower'], 'models/memory_lower_model.joblib')
    joblib.dump(bound_models['upper'], 'models/memory_upper_model.joblib')
    save_feature_list(feature_cols)

    print("\n✅ Models saved to models/")
    print("   memory_model.joblib")
    print("   memory_lower_model.joblib")
    print("   memory_upper_model.joblib")
    print("   memory_model_features.json")


if __name__ == '__main__':
    main()
