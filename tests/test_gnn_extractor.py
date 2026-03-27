import torch.nn as nn
from torchvision.models import resnet18

from blink.gnn_extractor import encode_layer, model_to_graph


def test_encode_conv_layer():
    conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    features = encode_layer(conv)
    assert len(features) == 12
    assert features[0] == 1.0

def test_encode_linear_layer():
    linear = nn.Linear(10, 5)
    features = encode_layer(linear)
    assert len(features) == 12
    assert features[1] == 1.0

def test_encode_norm_layer():
    norm = nn.BatchNorm2d(16)
    features = encode_layer(norm)
    assert len(features) == 12
    assert features[2] == 1.0

def test_model_to_graph_empty_model():
    model = nn.Module()
    graph = model_to_graph(model)
    assert graph.x.shape == (1, 12)
    assert graph.edge_index.shape[1] == 0

def test_model_to_graph_resnet18():
    model = resnet18(weights=None)
    graph = model_to_graph(model)
    assert graph.x.shape[0] > 10
    assert graph.x.shape[1] == 12
    assert graph.edge_index.shape[1] == graph.x.shape[0] - 1

def test_encode_layernorm():
    """Covers lines 53-56: LayerNorm uses normalized_shape branch"""
    ln = nn.LayerNorm(64)
    features = encode_layer(ln)
    assert len(features) == 12
    assert features[2] == 1.0  # 'Norm' one-hot
    assert features[7] == 64.0  # in_channels = normalized_shape[0]

def test_encode_groupnorm():
    """Covers lines 57-59: GroupNorm uses num_channels branch"""
    gn = nn.GroupNorm(num_groups=4, num_channels=16)
    features = encode_layer(gn)
    assert len(features) == 12
    assert features[2] == 1.0  # 'Norm' one-hot

def test_encode_pool_layer():
    """Covers lines 60-66: MaxPool2d branch"""
    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    features = encode_layer(pool)
    assert len(features) == 12
    assert features[3] == 1.0  # 'Pool' one-hot
    assert features[9] == 3.0  # kernel_size

def test_encode_multihead_attention():
    """Covers lines 67-70: MultiheadAttention branch"""
    mha = nn.MultiheadAttention(embed_dim=64, num_heads=8)
    features = encode_layer(mha)
    assert len(features) == 12
    assert features[4] == 1.0  # 'Attention' one-hot
    assert features[7] == 64.0  # in_channels = embed_dim
    assert features[9] == 8.0  # kernel_size = num_heads
