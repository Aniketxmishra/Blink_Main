from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from blink.gpu_predictor import GPUPredictor


class FakeModel:
    def __init__(self, output_val, is_log=True):
        self.output_val = output_val
        self.is_log = is_log
    def predict(self, X):
        if self.is_log:
            return np.ones(len(X)) * np.log1p(self.output_val)
        return np.ones(len(X)) * self.output_val

@pytest.fixture
def mock_joblib_load():
    with patch('joblib.load') as mock_load:
        def side_effect(path):
            path_str = str(path).lower()
            if 'lower' in path_str and 'memory' not in path_str:
                return FakeModel(10.0, is_log=True) 
            elif 'upper' in path_str and 'memory' not in path_str:
                return FakeModel(30.0, is_log=True) 
            elif 'memory' in path_str and 'lower' in path_str:
                return FakeModel(100.0, is_log=False)
            elif 'memory' in path_str and 'upper' in path_str:
                return FakeModel(300.0, is_log=False)
            elif 'memory' in path_str:
                return FakeModel(200.0, is_log=False)
            else:
                return FakeModel(20.0, is_log=True)
                
        mock_load.side_effect = side_effect
        yield mock_load

@pytest.fixture
def predictor(mock_joblib_load, tmp_path):
    with patch('os.path.exists', return_value=False): # mock json existence check
        with patch('blink.gpu_predictor._find_models_dir', return_value=str(tmp_path)):
            return GPUPredictor()

def test_predict_returns_dict(predictor):
    features = {'batch_size': 1, 'flops': 1e9, 'model_size_mb': 100}
    result = predictor.predict(features)
    assert isinstance(result, dict)
    assert 'exec_time_ms' in result

def test_predict_exec_time_positive(predictor):
    result = predictor.predict({'batch_size': 1})
    assert result['exec_time_ms'] > 0

def test_predict_memory_positive(predictor):
    result = predictor.predict({'batch_size': 1})
    assert result['memory_usage_mb'] > 0

def test_confidence_intervals_ordered(predictor):
    result = predictor.predict({'batch_size': 1})
    assert result['exec_time_lower'] <= result['exec_time_ms'] <= result['exec_time_upper']
    assert result['memory_lower_mb'] <= result['memory_usage_mb'] <= result['memory_upper_mb']

def test_batch_size_scaling(predictor):
    predictor.model.predict = MagicMock(side_effect=lambda df: np.log1p(df['batch_size'].values * 10.0))
    res1 = predictor.predict({'batch_size': 1})
    res16 = predictor.predict({'batch_size': 16})
    assert res16['exec_time_ms'] > res1['exec_time_ms']

def test_cache_hit_on_repeat_call(predictor):
    features = {'batch_size': 8, 'flops': 5e9}
    
    hits_before = predictor.cache_hits
    misses_before = predictor.cache_misses
    
    _ = predictor.predict(features) 
    assert predictor.cache_misses == misses_before + 1
    
    
    _ = predictor.predict(features)
    assert predictor.cache_hits == hits_before + 1

def test_predict_for_custom_model_tabular_fallback(predictor):
    features = {'batch_size': 1, 'flops': 1e9, 'model_size_mb': 100}
    res = predictor.predict_for_custom_model(features, batch_size=4)
    assert 'exec_time_ms' in res
    assert res['source'] == 'xgboost'

def test_predict_for_custom_model_invalid_input(predictor):
    with pytest.raises(ValueError):
        predictor.predict_for_custom_model("not_a_model_or_dict", batch_size=4)

def test_optimize_batch_size_valid(predictor):
    features = {'batch_size': 1, 'flops': 1e9, 'model_size_mb': 100}
    # Mock predict so it scales linearly enough not to exceed memory but still works
    predictor.model.predict = MagicMock(side_effect=lambda df: np.log1p(df['batch_size'].values * 10.0))
    
    res = predictor.optimize_batch_size(features, min_batch=1, max_batch=16, memory_limit_mb=8000)
    assert 'optimal_batch_size' in res
    assert 'pareto_front' in res
    assert len(res['batch_results']) > 0

def test_optimize_batch_size_memory_exceeded(predictor):
    features = {'batch_size': 1, 'flops': 1e9, 'model_size_mb': 10000}
    # Provide a memory limit of 100 (which it immediately exceeds)
    res = predictor.optimize_batch_size(features, min_batch=1, max_batch=16, memory_limit_mb=100)
    assert res['optimal_batch_size'] == 1
    assert res['error'] is not None

def test_compute_pareto_indices(predictor):
    # Dummy pareto front calculation check
    results = [
        {'throughput': 10, 'corrected_memory_mb': 10},  # pareto
        {'throughput': 8, 'corrected_memory_mb': 15},   # dominated
        {'throughput': 20, 'corrected_memory_mb': 50}   # pareto
    ]
    indices = predictor._compute_pareto_indices(results)
    assert 0 in indices
    assert 2 in indices
    assert 1 not in indices

