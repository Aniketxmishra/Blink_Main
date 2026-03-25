import pytest
import torch.nn as nn
from unittest.mock import patch, MagicMock
from blink import BlinkPredictor

@pytest.fixture
def mock_gpu_predictor():
    with patch('blink._predictor.BlinkPredictor._get_predictor') as mock_pred:
        instance = MagicMock()
        instance.predict.return_value = {
            "exec_time_ms": 15.0,
            "exec_time_lower": 10.0,
            "exec_time_upper": 20.0,
            "memory_usage_mb": 150.0,
            "memory_lower_mb": 100.0,
            "memory_upper_mb": 200.0,
            "source": "xgboost"
        }
        mock_pred.return_value = instance
        yield instance

@pytest.fixture
def mock_analyzer():
    with patch('blink.model_analyser.ModelAnalyzer') as mock_ana:
        instance = MagicMock()
        instance.extract_features.return_value = {'total_parameters': 1000}
        mock_ana.return_value = instance
        yield instance

def test_named_model_resnet18(mock_gpu_predictor, mock_analyzer):
    bp = BlinkPredictor()
    result = bp.predict("resnet18")
    assert result["exec_time_ms"] == 15.0
    mock_gpu_predictor.predict.assert_called_once()
    mock_analyzer.extract_features.assert_called_once()

def test_named_model_invalid_raises():
    bp = BlinkPredictor()
    with pytest.raises(ValueError, match="Unknown model 'invalid_model'"):
        bp.predict("invalid_model")

def test_nn_module_input(mock_gpu_predictor, mock_analyzer):
    bp = BlinkPredictor()
    model = nn.Linear(10, 2)
    result = bp.predict(model)
    assert result["exec_time_ms"] == 15.0

def test_predict_batch_returns_list(mock_gpu_predictor, mock_analyzer):
    bp = BlinkPredictor()
    batch_sizes = [1, 16, 32]
    results = bp.predict_batch("resnet18", batch_sizes=batch_sizes)
    assert isinstance(results, list)
    assert len(results) == 3

def test_result_keys_present(mock_gpu_predictor, mock_analyzer):
    bp = BlinkPredictor()
    result = bp.predict("resnet18")
    expected_keys = {
        "exec_time_ms", "exec_time_lower", "exec_time_upper",
        "memory_mb", "memory_lower_mb", "memory_upper_mb", "source"
    }
    assert expected_keys.issubset(result.keys())
