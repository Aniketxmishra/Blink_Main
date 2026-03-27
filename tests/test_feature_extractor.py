import pytest
import torch.nn as nn
from torchvision.models import mobilenet_v2, resnet18

from blink.feature_extractor import ModelFeatureExtractor


@pytest.fixture
def extractor(tmp_path):
    return ModelFeatureExtractor(save_dir=str(tmp_path))

def test_extract_returns_21_features(extractor):
    model = resnet18(weights=None)
    features = extractor.extract_model_features(model, input_shape=(3, 224, 224))
    
    assert isinstance(features, dict)
    assert "total_parameters" in features
    assert "flops" in features
    assert "model_depth" in features
    assert "num_conv_layers" in features

def test_flops_positive(extractor):
    model = mobilenet_v2(weights=None)
    features = extractor.extract_model_features(model, input_shape=(3, 224, 224))
    assert features["flops"] > 0

def test_param_count_matches_pytorch(extractor):
    model = resnet18(weights=None)
    actual_params = sum(p.numel() for p in model.parameters())
    
    features = extractor.extract_model_features(model, input_shape=(3, 224, 224))
    assert features["total_parameters"] == actual_params

def test_conv_layer_count_correct(extractor):
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    features = extractor.extract_model_features(model, input_shape=(3, 10, 10))
    assert features["num_conv_layers"] == 2
    assert features["num_fc_layers"] == 1

def test_depth_positive(extractor):
    model = mobilenet_v2(weights=None)
    features = extractor.extract_model_features(model, input_shape=(3, 224, 224))
    assert features["model_depth"] > 0

def test_process_all_profiling_data(extractor, tmp_path):
    import os

    import pandas as pd
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    df = pd.DataFrame({"model_name": ["resnet18"], "batch_size": [32], "exec_time_ms": [10.0], "memory_usage_mb": [100.0]})
    df.to_csv(csv_dir / "test.csv", index=False)
    
    extractor.process_all_profiling_data(str(csv_dir))
    assert os.path.exists(str(tmp_path / "features_test.csv"))
