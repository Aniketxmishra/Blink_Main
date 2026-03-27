"""
scripts/shap_explainer.py — SHAP-based prediction explanation for Blink

Wraps SHAP TreeExplainer around the Random Forest (exec time) and XGBoost
(memory) models to answer: "Why did the model predict THIS value?"

Usage (from project root):
    from scripts.shap_explainer import BlinkExplainer
    exp = BlinkExplainer()
    result = exp.explain_prediction(feature_dict, batch_size=32)
    # result contains shap_values, base_value, feature_names, friendly labels
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, 'models')

# ── Human-readable feature labels ────────────────────────────────────────────
EXEC_FEATURE_LABELS = {
    'batch_size':             'Batch Size',
    'flops':                  'Total FLOPs',
    'compute_memory_ratio':   'FLOPs / Model Size',
    'num_conv_layers':        'Conv Layers',
    'num_fc_layers':          'FC Layers',
    'num_bn_layers':          'BN Layers',
    'avg_conv_kernel_size':   'Avg Kernel Size',
    'max_conv_channels':      'Max Conv Channels',
    'total_conv_params':      'Conv Parameters',
    'total_fc_params':        'FC Parameters',
    'model_depth':            'Model Depth',
    'model_size_mb':          'Model Size (MB)',
}

EXEC_FEATURE_COLS = list(EXEC_FEATURE_LABELS.keys())


class BlinkExplainer:
    """
    Wraps SHAP TreeExplainer around Blink's exec-time and memory models.
    Provides both per-prediction (waterfall) and global (bar) explanations.
    """

    def __init__(self):
        import shap as _shap
        self.shap = _shap

        # Execution time model (Random Forest)
        exec_path = os.path.join(MODEL_PATH, 'random_forest_model.joblib')
        self.exec_model = joblib.load(exec_path)
        self._exec_explainer = None  # lazy-init (slow first build)

        # Memory model (XGBoost)
        mem_path = os.path.join(MODEL_PATH, 'memory_model.joblib')
        self.mem_model = joblib.load(mem_path)
        self._mem_explainer = None

        # Memory feature list
        import json
        feat_json = os.path.join(MODEL_PATH, 'memory_model_features.json')
        if os.path.exists(feat_json):
            with open(feat_json) as f:
                self.mem_feature_cols = json.load(f)['features']
        else:
            self.mem_feature_cols = EXEC_FEATURE_COLS

    # ── Explainers (lazy, cached) ─────────────────────────────────────────────

    def _get_exec_explainer(self):
        if self._exec_explainer is None:
            self._exec_explainer = self.shap.TreeExplainer(self.exec_model)
        return self._exec_explainer

    def _get_mem_explainer(self):
        if self._mem_explainer is None:
            self._mem_explainer = self.shap.TreeExplainer(self.mem_model)
        return self._mem_explainer

    # ── Core explainability ───────────────────────────────────────────────────

    def explain_exec(self, feature_dict: dict, batch_size: int = 32) -> dict:
        """
        Compute SHAP values for a single execution-time prediction.

        Returns dict with:
          shap_values   : np.array, one value per feature
          base_value    : float, model's expected output (log scale)
          prediction    : float, model's prediction (log scale)
          feature_names : list[str]
          feature_labels: list[str]  (human-readable)
          feature_values: list[float]
        """
        fd = dict(feature_dict)
        fd['batch_size'] = batch_size

        row = pd.DataFrame([{k: fd.get(k, 0) for k in EXEC_FEATURE_COLS}])
        explainer = self._get_exec_explainer()
        sv = explainer.shap_values(row)

        # RandomForest may return list-of-arrays for multi-output; take [0]
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv).flatten()

        return {
            'shap_values':    sv,
            'base_value':     float(explainer.expected_value
                                    if not hasattr(explainer.expected_value, '__len__')
                                    else explainer.expected_value[0]),
            'prediction':     float(self.exec_model.predict(row)[0]),
            'feature_names':  EXEC_FEATURE_COLS,
            'feature_labels': [EXEC_FEATURE_LABELS.get(k, k) for k in EXEC_FEATURE_COLS],
            'feature_values': [float(row[k].iloc[0]) for k in EXEC_FEATURE_COLS],
            'target_unit':    'log(exec_time_ms)',
        }

    def explain_memory(self, feature_dict: dict, batch_size: int = 32) -> dict:
        """
        Compute SHAP values for a single memory prediction.
        """
        fd = dict(feature_dict)
        fd['batch_size'] = batch_size
        if fd.get('weight_memory_mb') is None:
            fd['weight_memory_mb'] = fd.get('model_size_mb', 0)
        if fd.get('activation_memory_mb') is None:
            fd['activation_memory_mb'] = fd.get('model_size_mb', 0) * 0.5 * batch_size
        if fd.get('activation_memory_per_sample') is None:
            fd['activation_memory_per_sample'] = fd['activation_memory_mb'] / max(batch_size, 1)
        if fd.get('flops_per_activation_mb') is None:
            fd['flops_per_activation_mb'] = fd.get('flops', 0) / max(fd['activation_memory_mb'], 1e-3)
        if fd.get('input_resolution_factor') is None:
            fd['input_resolution_factor'] = 224 * 224

        feat_cols = self.mem_feature_cols
        row = pd.DataFrame([{k: fd.get(k, 0) for k in feat_cols}])
        explainer = self._get_mem_explainer()
        sv = explainer.shap_values(row)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv).flatten()

        labels = {
            'activation_memory_mb':         'Activation Memory (MB)',
            'weight_memory_mb':             'Weight Memory (MB)',
            'activation_memory_per_sample': 'Activation / Sample',
            'flops_per_activation_mb':      'FLOPs per Activation MB',
            'input_resolution_factor':      'Input Resolution',
            **EXEC_FEATURE_LABELS,
        }

        return {
            'shap_values':    sv,
            'base_value':     float(explainer.expected_value
                                    if not hasattr(explainer.expected_value, '__len__')
                                    else explainer.expected_value[0]),
            'prediction':     float(self.mem_model.predict(row)[0]),
            'feature_names':  feat_cols,
            'feature_labels': [labels.get(k, k) for k in feat_cols],
            'feature_values': [float(row[k].iloc[0]) for k in feat_cols],
            'target_unit':    'peak_memory_mb',
        }

    # ── Global feature importance ─────────────────────────────────────────────

    def global_importance_exec(self, n_samples: int = 50) -> pd.DataFrame:
        """
        Compute mean |SHAP| across a small synthetic grid of feature values
        to give an estimate of global feature importance for exec time.
        Uses the training dataset if available, else a simplified grid.
        """
        data_path = os.path.join(ROOT, 'data', 'enriched', 'enriched_data.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            avail = [c for c in EXEC_FEATURE_COLS if c in df.columns]
            X = df[avail].fillna(0).head(n_samples)
            # Pad with zeros for missing columns
            for c in EXEC_FEATURE_COLS:
                if c not in X.columns:
                    X[c] = 0
            X = X[EXEC_FEATURE_COLS]
        else:
            rng = np.random.default_rng(42)
            X = pd.DataFrame(rng.standard_normal((n_samples, len(EXEC_FEATURE_COLS))),
                             columns=EXEC_FEATURE_COLS)

        explainer = self._get_exec_explainer()
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]

        mean_abs = np.abs(sv).mean(axis=0)
        df_imp = pd.DataFrame({
            'feature': EXEC_FEATURE_COLS,
            'label':   [EXEC_FEATURE_LABELS.get(k, k) for k in EXEC_FEATURE_COLS],
            'importance': mean_abs,
        }).sort_values('importance', ascending=False)
        return df_imp


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    exp = BlinkExplainer()
    test_feat = {
        'batch_size': 32, 'flops': 1.8e9, 'compute_memory_ratio': 20.0,
        'num_conv_layers': 16, 'num_fc_layers': 1, 'num_bn_layers': 16,
        'avg_conv_kernel_size': 3.0, 'max_conv_channels': 512,
        'total_conv_params': 11e6, 'total_fc_params': 1000, 'model_depth': 18,
        'model_size_mb': 44.7,
    }
    r = exp.explain_exec(test_feat, batch_size=32)
    print("SHAP values:", dict(zip(r['feature_labels'], r['shap_values'].round(4))))
    print("Base value:", r['base_value'])
    print("Prediction:", r['prediction'])
