import pytest
import torch.nn as nn
from blink.model_analyser import ModelAnalyzer


def test_model_analyser_direct():
    analyzer = ModelAnalyzer()
    info = analyzer.extract_features(nn.Linear(10, 2), input_shape=(1, 10))
    assert 'total_parameters' in info


def test_analyze_unsupported_model():
    analyzer = ModelAnalyzer()
    info = analyzer.extract_features(nn.Linear(10, 2), input_shape=(1, 10))
    assert 'total_parameters' in info


def test_analyze_conv_fallback():
    analyzer = ModelAnalyzer()
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 2)
    )
    info = analyzer.extract_features(model, input_shape=(3, 32, 32))
    assert info['total_parameters'] > 0
    assert info['flops'] > 0

def test_extract_llm_features_via_mock():
    """Covers lines 175-222: _extract_llm_features path"""
    from unittest.mock import MagicMock, patch

    analyzer = ModelAnalyzer()

    # Build a tiny real nn.Module so param/buffer iteration works
    inner_model = nn.Linear(16, 16)

    # Attach a mock config — mimics HuggingFace PreTrainedModel.config
    mock_cfg = MagicMock()
    mock_cfg.vocab_size = 1000
    mock_cfg.hidden_size = 64
    mock_cfg.num_hidden_layers = 2
    mock_cfg.num_attention_heads = 4
    mock_cfg.intermediate_size = 256
    inner_model.config = mock_cfg

    # Patch _is_huggingface_llm to return True
    with patch.object(analyzer, '_is_huggingface_llm', return_value=True):
        info = analyzer.extract_features(inner_model, batch_size=2, seq_len=32)

    assert info['is_llm'] is True
    assert info['vocab_size'] == 1000
    assert info['hidden_size'] == 64
    assert info['num_hidden_layers'] == 2
    assert info['kv_cache_size_mb'] > 0
    assert info['flops'] > 0

def test_interval_pinball_loss():
    """Covers lines 438-446: ModelAnalyzer.interval_pinball_loss"""
    import numpy as np
    y_true = np.array([10.0, 20.0, 30.0])
    q_lower = np.array([8.0, 18.0, 28.0])
    q_upper = np.array([12.0, 22.0, 32.0])
    loss = ModelAnalyzer.interval_pinball_loss(y_true, q_lower, q_upper)
    # All predictions within bounds ⟹ loss = mean of widths (4.0 each)
    assert abs(loss - 4.0) < 1e-6

def test_interval_pinball_loss_with_violations():
    """Covers penalty branches in interval_pinball_loss"""
    import numpy as np
    y_true = np.array([5.0, 25.0])    # 5 < lower, 25 > upper
    q_lower = np.array([8.0, 18.0])
    q_upper = np.array([12.0, 22.0])
    loss = ModelAnalyzer.interval_pinball_loss(y_true, q_lower, q_upper)
    assert loss > 4.0  # must be larger than plain width due to penalties

def test_ood_split_batch_size():
    """Covers lines 455-458: ood_extrapolation_split batch_size mode"""
    import pandas as pd
    df = pd.DataFrame({
        'model_name': ['a', 'b', 'c', 'd'],
        'batch_size': [1, 4, 16, 32],
        'exec_time_ms': [10, 15, 25, 40]
    })
    train, test = ModelAnalyzer.ood_extrapolation_split(df, mode='batch_size', train_thresh=8)
    assert all(train['batch_size'] <= 8)
    assert all(test['batch_size'] > 8)

def test_ood_split_family():
    """Covers lines 460-465: ood_extrapolation_split family mode"""
    import pandas as pd
    rows = [{'model_name': f'resnet_{i}', 'batch_size': i, 'exec_time_ms': i*10} for i in range(1, 20)]
    rows += [{'model_name': f'vgg_{i}', 'batch_size': i, 'exec_time_ms': i*5} for i in range(1, 20)]
    df = pd.DataFrame(rows)
    train, test = ModelAnalyzer.ood_extrapolation_split(df, mode='family')
    assert len(train) + len(test) == len(df)
