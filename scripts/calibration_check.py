"""
scripts/calibration_check.py  —  Priority 2: Interval Calibration

Checks whether the 80% confidence intervals produced by NeuSight's
quantile regression models actually contain 80% of real measurements.

Generates:
  results/calibration_reliability.png  — reliability diagram
  results/calibration_report.txt       — numeric summary

Usage:
  python scripts/calibration_check.py
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH  = 'data/enriched/enriched_data.csv'
MODEL_PATH = 'models'
RESULTS    = 'results'
os.makedirs(RESULTS, exist_ok=True)


# ── Feature columns (keep in sync with prediction_model.py) ─────────────────
EXEC_FEATURES = [
    'batch_size', 'flops', 'compute_memory_ratio',
    'num_conv_layers', 'num_fc_layers', 'num_bn_layers',
    'avg_conv_kernel_size', 'max_conv_channels',
    'total_conv_params', 'total_fc_params', 'model_depth', 'model_size_mb'
]

import json
MEM_FEAT_JSON = os.path.join(MODEL_PATH, 'memory_model_features.json')
if os.path.exists(MEM_FEAT_JSON):
    with open(MEM_FEAT_JSON) as f:
        MEM_FEATURES = json.load(f)['features']
else:
    MEM_FEATURES = EXEC_FEATURES  # fallback


def load_data():
    df = pd.read_csv(DATA_PATH)
    if 'timing_cv' in df.columns:
        df = df[df['timing_cv'] <= 0.15]
    df = df[df['peak_memory_mb'] > 0].copy()
    if 'activation_memory_mb' in df.columns and 'batch_size' in df.columns:
        df['activation_memory_per_sample'] = df['activation_memory_mb'] / df['batch_size'].clip(1)
    if 'weight_memory_mb' not in df.columns:
        df['weight_memory_mb'] = df['model_size_mb']
    if 'flops_per_activation_mb' not in df.columns and 'activation_memory_mb' in df.columns:
        df['flops_per_activation_mb'] = df['flops'] / df['activation_memory_mb'].clip(1e-3)
    if 'input_resolution_factor' not in df.columns:
        df['input_resolution_factor'] = 224 * 224
    return df


def compute_coverage(y_true, y_lower, y_upper):
    """Fraction of true values that fall inside [lower, upper]."""
    inside = ((y_true >= y_lower) & (y_true <= y_upper))
    return inside.mean(), inside


def calibration_report(df, feat_cols, y_col, lower_model, upper_model, label):
    """Return coverage and average interval width."""
    avail = [c for c in feat_cols if c in df.columns]
    X = df[avail].fillna(0)
    y = df[y_col].values

    y_lower = lower_model.predict(X[avail])
    y_upper = upper_model.predict(X[avail])

    coverage, inside = compute_coverage(y, y_lower, y_upper)
    avg_width = np.mean(y_upper - y_lower)

    print(f"\n{label}")
    print(f"  Nominal coverage : 80%")
    print(f"  Actual coverage  : {coverage*100:.1f}%")
    print(f"  Avg CI width     : {avg_width:.1f} units")
    print(f"  Calibration gap  : {abs(coverage - 0.80)*100:.1f}%  "
          f"({'over-confident' if coverage < 0.80 else 'conservative'})")

    return coverage, avg_width, y, y_lower, y_upper


def plot_reliability_diagram(results, save_path):
    """
    Reliability diagram showing:
      - Actual vs nominal coverage for each model/target
      - Interval widths
      - Per-architecture coverage breakdown (exec time)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('NeuSight — 80% Confidence Interval Calibration (Priority 2)',
                 fontsize=13, fontweight='bold')

    targets  = [r['label'] for r in results]
    coverages= [r['coverage'] for r in results]
    widths   = [r['avg_width'] for r in results]
    colors   = ['#2ecc71' if abs(c - 0.80) < 0.05 else '#e74c3c' for c in coverages]

    # ── Panel 1: Coverage bars ────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(targets, [c * 100 for c in coverages], color=colors, width=0.5)
    ax.axhline(80, color='navy', linestyle='--', linewidth=1.5, label='Nominal 80%')
    ax.axhspan(75, 85, alpha=0.08, color='green', label='±5% acceptable')
    ax.set_ylabel('Actual Coverage (%)')
    ax.set_title('CI Coverage (target = 80%)')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, cov in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{cov*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── Panel 2: Predicted vs Actual scatter (exec time) ─────────────────────
    ax = axes[1]
    exec_res = next(r for r in results if 'Exec' in r['label'])
    y, yl, yu = exec_res['y'], exec_res['y_lower'], exec_res['y_upper']
    inside = (y >= yl) & (y <= yu)
    ax.scatter(y[inside],  y[inside],  color='#2ecc71', s=20, alpha=0.7, label='Inside CI')
    ax.scatter(y[~inside], y[~inside], color='#e74c3c', s=30, alpha=0.9, label='Outside CI',
               marker='x', linewidths=1.5)
    lo = y.min(); hi = y.max()
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], alpha=0.05, color='green')
    ax.set_xlabel('Actual Exec Time (ms)')
    ax.set_ylabel('Position on diagonal')
    ax.set_title(f'Exec Time: {inside.mean()*100:.1f}% inside 80% CI')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Interval width vs actual value (exec time) ──────────────────
    ax = axes[2]
    widths_per_point = yu - yl
    sc = ax.scatter(y, widths_per_point, c=inside.astype(float),
                    cmap='RdYlGn', s=30, alpha=0.8, vmin=0, vmax=1)
    ax.set_xlabel('Actual Exec Time (ms)')
    ax.set_ylabel('CI Width (ms)')
    ax.set_title('CI Width vs Actual Value')
    ax.grid(True, alpha=0.3)
    green_patch = mpatches.Patch(color='green', label='Inside CI')
    red_patch   = mpatches.Patch(color='red',   label='Outside CI')
    ax.legend(handles=[green_patch, red_patch], fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {save_path}")


def main():
    print("=" * 60)
    print("NeuSight — Confidence Interval Calibration (Priority 2)")
    print("=" * 60)

    df = load_data()
    print(f"Loaded {len(df)} rows across {df['model_name'].nunique()} architectures")

    results = []

    # ── Execution time CI ─────────────────────────────────────────────────────
    exec_lower = joblib.load(os.path.join(MODEL_PATH, 'execution_lower_model.joblib'))
    exec_upper = joblib.load(os.path.join(MODEL_PATH, 'execution_upper_model.joblib'))

    avail_exec = [c for c in EXEC_FEATURES if c in df.columns]
    # execution_time_ms is on log scale in the model
    df['log_exec'] = np.log1p(df['execution_time_ms'])

    cov, width, y, yl, yu = calibration_report(
        df, EXEC_FEATURES, 'log_exec', exec_lower, exec_upper,
        'Execution Time CI (log scale)'
    )
    # Convert back to ms scale for the plot
    results.append({
        'label': 'Exec Time',
        'coverage': cov,
        'avg_width': np.expm1(width),
        'y':       np.expm1(y),
        'y_lower': np.expm1(yl),
        'y_upper': np.expm1(yu),
    })

    # ── Memory CI ─────────────────────────────────────────────────────────────
    mem_lower = joblib.load(os.path.join(MODEL_PATH, 'memory_lower_model.joblib'))
    mem_upper = joblib.load(os.path.join(MODEL_PATH, 'memory_upper_model.joblib'))

    mem_avail = [c for c in MEM_FEATURES if c in df.columns]
    cov_m, width_m, y_m, yl_m, yu_m = calibration_report(
        df, MEM_FEATURES, 'peak_memory_mb', mem_lower, mem_upper,
        'Memory CI (MB)'
    )
    results.append({
        'label': 'Memory',
        'coverage': cov_m,
        'avg_width': width_m,
        'y': y_m, 'y_lower': yl_m, 'y_upper': yu_m,
    })

    # ── Write text report ─────────────────────────────────────────────────────
    report_path = os.path.join(RESULTS, 'calibration_report.txt')
    with open(report_path, 'w') as f:
        f.write("NeuSight Calibration Report\n")
        f.write("=" * 40 + "\n")
        for r in results:
            status = 'WELL CALIBRATED' if abs(r['coverage'] - 0.80) < 0.05 else \
                     'OVER-CONFIDENT' if r['coverage'] < 0.80 else 'CONSERVATIVE'
            f.write(f"\n{r['label']}:\n")
            f.write(f"  Coverage  : {r['coverage']*100:.1f}%  ({status})\n")
            f.write(f"  Avg width : {r['avg_width']:.1f}\n")
    print(f"  Report:  {report_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(RESULTS, 'calibration_reliability.png')
    plot_reliability_diagram(results, plot_path)

    print("\nCalibration Summary:")
    for r in results:
        status = '[OK]' if abs(r['coverage'] - 0.80) < 0.05 else '[WARN]️'
        print(f"  {status} {r['label']}: {r['coverage']*100:.1f}% "
              f"(nominal 80%, width={r['avg_width']:.1f})")


if __name__ == '__main__':
    main()
