import joblib
import pandas as pd
import numpy as np
import os
import torch
from datetime import datetime

class GPUPredictor:
    """Scalable GPU usage prediction system with caching and batch processing"""
    
    def __init__(self, model_path='models/random_forest_model.joblib', 
                 memory_model_path='models/memory_model.joblib', cache_size=100):
        self.model = joblib.load(model_path)
        try:
            self.memory_model = joblib.load(memory_model_path)
            self.memory_model_lower = joblib.load('models/memory_lower_model.joblib')
            self.memory_model_upper = joblib.load('models/memory_upper_model.joblib')
            self.has_memory_model = True
            # Load activation-aware feature list saved during training
            import json
            mem_feat_path = 'models/memory_model_features.json'
            if os.path.exists(mem_feat_path):
                with open(mem_feat_path) as f:
                    self.memory_feature_cols = json.load(f)['features']
                print(f"Loaded memory model v2 ({len(self.memory_feature_cols)} features)")
            else:
                # Fallback to old feature set if JSON not present
                self.memory_feature_cols = [
                    'batch_size', 'flops', 'compute_memory_ratio',
                    'num_conv_layers', 'num_fc_layers', 'num_bn_layers',
                    'avg_conv_kernel_size', 'max_conv_channels',
                    'total_conv_params', 'total_fc_params', 'model_depth', 'model_size_mb'
                ]
        except FileNotFoundError:
            print(f"Warning: Memory model not found at {memory_model_path}. Falling back to heuristic.")
            self.has_memory_model = False
            self.memory_model_lower = None
            self.memory_model_upper = None
            self.memory_feature_cols = []
            
        try:
            self.model_lower = joblib.load('models/execution_lower_model.joblib')
            self.model_upper = joblib.load('models/execution_upper_model.joblib')
            self.has_exec_bounds = True
        except FileNotFoundError:
            print(f"Warning: Execution bound models not found. Intervals will not be available.")
            self.has_exec_bounds = False
            self.model_lower = None
            self.model_upper = None
            
        from pathlib import Path
        if Path("models/gnn_predictor.pth").exists():
            from gnn_model import ArchitectureGNN
            import torch
            
            self.use_gnn = True
            self.gnn_model = ArchitectureGNN()
            self.gnn_model.load_state_dict(torch.load("models/gnn_predictor.pth", map_location='cpu'))
            self.gnn_model.eval()
            print("Loaded ArchitectureGNN from models/gnn_predictor.pth")
        else:
            self.use_gnn = False
            
        self.prediction_cache = {}  # Cache for fast repeated predictions
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        # Define the feature order used during training (matches prediction_model.py)
        self.feature_cols = [
            'batch_size',
            'flops',
            'compute_memory_ratio',
            'num_conv_layers',
            'num_fc_layers',
            'num_bn_layers',
            'avg_conv_kernel_size',
            'max_conv_channels',
            'total_conv_params',
            'total_fc_params',
            'model_depth',
            'model_size_mb'
        ]
    
    def predict(self, features_batch):
        """Make predictions for a batch of models efficiently"""
        if not isinstance(features_batch, list):
            features_batch = [features_batch]
            
        results = []
        features_to_predict = []
        cache_indices = []
        
        # Check cache first
        for i, features in enumerate(features_batch):
            cache_key = self._get_cache_key(features)
            if cache_key in self.prediction_cache:
                results.append(self.prediction_cache[cache_key])
                self.cache_hits += 1
            else:
                results.append(None)
                features_to_predict.append(features)
                cache_indices.append(i)
                self.cache_misses += 1
        
        # Make predictions for cache misses
        if features_to_predict:
            # Extract features in the exact order used during training
            numeric_features = []
            for features in features_to_predict:
                feature_dict = {}
                for col in self.feature_cols:
                    # Default batch size 1, others 0 if missing
                    default_val = 1 if col == 'batch_size' else 0
                    feature_dict[col] = features.get(col, default_val)
                numeric_features.append(feature_dict)
            
            # Convert to DataFrame with specific column order
            features_df = pd.DataFrame(numeric_features)[self.feature_cols]
            
            # Make batch prediction (on log scale)
            predictions_log = self.model.predict(features_df)
            
            # Transform back to original scale
            predictions = np.expm1(predictions_log)
            
            # Bound models
            if self.has_exec_bounds:
                lower_bounds_log = self.model_lower.predict(features_df)
                upper_bounds_log = self.model_upper.predict(features_df)
                lower_bounds = np.expm1(lower_bounds_log)
                upper_bounds = np.expm1(upper_bounds_log)
            else:
                lower_bounds = predictions
                upper_bounds = predictions
            
            # Predict memory usage using the activation-aware feature set
            if self.has_memory_model:
                mem_numeric = []
                for features in features_to_predict:
                    mem_row = {
                        col: features.get(col, 0)
                        for col in self.memory_feature_cols
                    }
                    # batch_size default = 1 if missing
                    if 'batch_size' in self.memory_feature_cols:
                        mem_row['batch_size'] = features.get('batch_size', 1)
                    mem_numeric.append(mem_row)
                mem_df = pd.DataFrame(mem_numeric)[self.memory_feature_cols]
                memory_preds   = self.memory_model.predict(mem_df)
                memory_lower_p = self.memory_model_lower.predict(mem_df)
                memory_upper_p = self.memory_model_upper.predict(mem_df)
            else:
                memory_preds = memory_lower_p = memory_upper_p = [None] * len(predictions)
            
            # Update results and cache
            for i, pred_idx in enumerate(cache_indices):
                pred_dict = {
                    'exec_time_ms': float(max(1.0, predictions[i])),
                    'exec_time_lower': float(max(1.0, lower_bounds[i])),
                    'exec_time_upper': float(max(predictions[i], upper_bounds[i])),
                    'memory_usage_mb': float(max(10.0, memory_preds[i])) if self.has_memory_model else None,
                    'memory_lower_mb': float(max(10.0, memory_lower_p[i])) if self.has_memory_model else None,
                    'memory_upper_mb': float(max(memory_preds[i], memory_upper_p[i])) if self.has_memory_model else None
                }
                
                results[pred_idx] = pred_dict
                
                # Update cache
                cache_key = self._get_cache_key(features_batch[pred_idx])
                self.prediction_cache[cache_key] = pred_dict
            
            # Limit cache size
            if len(self.prediction_cache) > self.cache_size:
                # Remove oldest entries (simple approach)
                keys_to_remove = list(self.prediction_cache.keys())[:-self.cache_size]
                for key in keys_to_remove:
                    del self.prediction_cache[key]
        
        return results[0] if len(results) == 1 else results
        
    def predict_for_custom_model(self, model, batch_size):
        import torch
        import torch.nn as nn
        from feature_extractor import ModelFeatureExtractor
        from gnn_extractor import model_to_graph
        
        # Determine fallback path first
        needs_fallback = True
        result = {}
        
        if getattr(self, 'use_gnn', False) and isinstance(model, nn.Module):
            try:
                device = next(model.parameters()).device
                model.to("cpu")
                graph = model_to_graph(model)
                model.to(device)
                
                with torch.no_grad():
                    bs_tensor = torch.tensor([[batch_size]], dtype=torch.float32)
                    out = self.gnn_model(graph, bs_tensor)
                
                exec_time = float(np.expm1(out[0, 0].item()))
                memory = float(np.expm1(out[0, 1].item()))
                
                # We don't have quantile predictions for GNN, use standard 10% bounds
                result = {
                    'exec_time_ms': exec_time,
                    'memory_usage_mb': memory,
                    'exec_lower_ms': exec_time * 0.9,
                    'exec_upper_ms': exec_time * 1.1,
                    'memory_lower_mb': memory * 0.9,
                    'memory_upper_mb': memory * 1.1,
                    'source': 'gnn'
                }
                needs_fallback = False
            except Exception as e:
                print(f"GNN Prediction Failed: {e}. Falling back to XGBoost tabular path.")

        if needs_fallback:
            # Fall back to tabular prediction
            extractor = ModelFeatureExtractor(save_dir='data/temp')
            
            if isinstance(model, nn.Module):
                # Pass a dummy input shape, (3, 224, 224) is safe for CNNs
                device = next(model.parameters()).device
                model.to("cpu")
                model_features = extractor.extract_model_features(model, (3, 224, 224))
                model.to(device)
            elif isinstance(model, dict):
                model_features = model
            else:
                raise ValueError("Model must be an nn.Module or feature dictionary")
                
            model_features['batch_size'] = batch_size
            pred_payload = self.predict(model_features)
            
            result = {
                'exec_time_ms': pred_payload['exec_time_ms'],
                'memory_usage_mb': pred_payload['memory_usage_mb'],
                'exec_lower_ms': pred_payload['exec_time_lower'],
                'exec_upper_ms': pred_payload['exec_time_upper'],
                'memory_lower_mb': pred_payload['memory_lower_mb'],
                'memory_upper_mb': pred_payload['memory_upper_mb'],
                'source': 'xgboost'
            }
            
        return result
    
    def _get_cache_key(self, features):
        """Generate a cache key from features"""
        key_parts = []
        for k in self.feature_cols:  # Use the same order as feature columns
            if k in features:
                key_parts.append(f"{k}:{features[k]}")
        return "|".join(key_parts)
    
    def get_cache_stats(self):
        """Return cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_size": len(self.prediction_cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }

    def optimize_batch_size(self, model_features, min_batch=1, max_batch=128, memory_limit_mb=8000):
        """Find the efficient optimal batch size using the knee/elbow method.

        Raw argmax(throughput) always returns max_batch because GPU execution time
        scales sub-linearly with batch size (GPU parallelism means bigger batches
        are proportionally faster). The elbow/knee method finds the inflection point
        on the throughput curve — the batch size where marginal gains start to
        plateau — which is the true engineering optimum.
        """
        # Build dense candidate set: powers-of-2 + intermediate midpoints
        candidates = set([min_batch, max_batch])
        b = 1
        while b <= max_batch:
            if b >= min_batch:
                candidates.add(b)
                mid = b * 3 // 2
                if min_batch <= mid <= max_batch:
                    candidates.add(mid)
            b *= 2
        # Also add ~20 linearly-spaced points for smooth curve
        step = max(1, (max_batch - min_batch) // 20)
        for v in range(min_batch, max_batch + 1, step):
            candidates.add(v)
        candidates = sorted(candidates)

        batch_results = []

        for batch_size in candidates:
            if batch_size < 1:
                continue
            features = model_features.copy()
            features['batch_size'] = batch_size

            # ── Scale activation memory with batch size ──────────────────────
            # activation_memory_mb was estimated at batch_size=1 during feature
            # extraction. The memory model needs the correct value for each
            # candidate batch size, or memory stays flat and the optimizer
            # always picks min_batch.
            act_per_sample = model_features.get('activation_memory_per_sample')
            if act_per_sample is None and model_features.get('activation_memory_mb'):
                # Derive per-sample cost from base extraction (assumed batch=1)
                act_per_sample = model_features['activation_memory_mb']
            if act_per_sample:
                features['activation_memory_mb'] = act_per_sample * batch_size
                features['activation_memory_per_sample'] = act_per_sample
            # weight_memory_mb doesn't scale with batch
            features['weight_memory_mb'] = model_features.get('weight_memory_mb',
                                            model_features.get('model_size_mb', 0))

            prediction_payload = self.predict(features)
            exec_time = max(prediction_payload['exec_time_ms'], 0.001)
            exec_lower = prediction_payload['exec_time_lower']
            exec_upper = prediction_payload['exec_time_upper']

            if self.has_memory_model:
                memory_usage = prediction_payload['memory_usage_mb']
                memory_lower = prediction_payload['memory_lower_mb']
                memory_upper = prediction_payload['memory_upper_mb']
            else:
                base_memory = model_features.get('model_size_mb', 0) or 0
                mem_scale = 0.5 if model_features.get('total_parameters', 0) > 100_000_000 else 0.3
                memory_usage = base_memory + (base_memory * mem_scale * batch_size)
                memory_lower = memory_usage * 0.9
                memory_upper = memory_usage * 1.1

            if memory_usage > memory_limit_mb:
                continue

            throughput = (batch_size * 1000.0) / exec_time

            batch_results.append({
                'batch_size': batch_size,
                'exec_time_ms': exec_time,
                'exec_lower_ms': exec_lower,
                'exec_upper_ms': exec_upper,
                'throughput': throughput,
                'memory_usage_mb': memory_usage,
                'memory_lower_mb': memory_lower,
                'memory_upper_mb': memory_upper
            })

        if not batch_results:
            return {
                'optimal_batch_size': min_batch,
                'predicted_execution_time': None,
                'exec_lower_ms': None,
                'exec_upper_ms': None,
                'estimated_memory_usage': None,
                'memory_lower_mb': None,
                'memory_upper_mb': None,
                'batch_results': [],
                'error': 'All batch sizes exceed the memory limit. Increase memory limit or reduce max batch size.'
            }

        batch_results.sort(key=lambda x: x['batch_size'])

        # ── Activation-memory corrected efficiency ──────────────────────────
        # Problem: the trained memory model predicts near-flat memory vs batch
        # size (it only learned model-weight memory, not activation memory).
        # Activation memory scales linearly: ~2 * params * bytes_per_element
        # per sample in the forward pass.
        #
        # We estimate activation_mb_per_sample from model_size_mb:
        #   activations ≈ 50% of param memory per sample (empirical for CNNs)
        # Then: corrected_memory = max(predicted_memory, base + bs * act_cost)
        # This forces realistic memory growth even when the model under-predicts.

        model_size_mb = model_features.get('model_size_mb', 0) or 0
        # Activation heuristic: ~0.5 MB per sample per MB of param memory
        # (conservative lower bound; real activation memory can be 2-5x this)
        activation_cost_per_sample = max(model_size_mb * 0.5, 1.0)  # MB / sample

        base_mem = batch_results[0]['memory_usage_mb'] if batch_results else 0

        for r in batch_results:
            bs = r['batch_size']
            predicted_mem = r['memory_usage_mb']
            # Corrected = at least base + linear activation growth
            activation_mem = base_mem + activation_cost_per_sample * (bs - batch_results[0]['batch_size'])
            corrected_mem  = max(predicted_mem, activation_mem, 1.0)
            r['corrected_memory_mb'] = corrected_mem
            r['efficiency'] = r['throughput'] / corrected_mem

        optimal_row = max(batch_results, key=lambda r: r['efficiency'])

        # ── Pareto front (Priority 4) ────────────────────────────────────────
        # Non-dominated set in (throughput ↑, corrected_memory ↓) space.
        # A point is Pareto-optimal if no other point has both higher throughput
        # AND lower memory — i.e. any improvement in one metric costs the other.
        pareto_indices = self._compute_pareto_indices(batch_results)
        for i, r in enumerate(batch_results):
            r['is_pareto'] = (i in pareto_indices)

        return {
            'optimal_batch_size': optimal_row['batch_size'],
            'predicted_execution_time': optimal_row['exec_time_ms'],
            'exec_lower_ms': optimal_row['exec_lower_ms'],
            'exec_upper_ms': optimal_row['exec_upper_ms'],
            'estimated_memory_usage': optimal_row['memory_usage_mb'],
            'memory_lower_mb': optimal_row['memory_lower_mb'],
            'memory_upper_mb': optimal_row['memory_upper_mb'],
            'batch_results': batch_results,
            'pareto_front': [batch_results[i] for i in sorted(pareto_indices)],
        }

    @staticmethod
    def _compute_pareto_indices(batch_results):
        """Return indices of Pareto-optimal points in (throughput↑, memory↓) space."""
        n = len(batch_results)
        is_pareto = [True] * n
        for i in range(n):
            ti = batch_results[i]['throughput']
            mi = batch_results[i].get('corrected_memory_mb',
                                      batch_results[i].get('memory_usage_mb', 1.0))
            for j in range(n):
                if i == j:
                    continue
                tj = batch_results[j]['throughput']
                mj = batch_results[j].get('corrected_memory_mb',
                                           batch_results[j].get('memory_usage_mb', 1.0))
                # j dominates i if it has >= throughput AND <= memory, strictly one
                if tj >= ti and mj <= mi and (tj > ti or mj < mi):
                    is_pareto[i] = False
                    break
        return {i for i, v in enumerate(is_pareto) if v}
