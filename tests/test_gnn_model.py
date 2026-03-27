import pytest
import torch
from torch_geometric.data import Data

from blink.gnn_model import ArchitectureGNN


@pytest.fixture
def dummy_graph():
    x = torch.randn(2, 12)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

@pytest.fixture
def another_dummy_graph():
    x = torch.randn(3, 12)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def test_forward_pass_shape(dummy_graph):
    model = ArchitectureGNN(node_feature_dim=12, hidden_dim=64)
    bs = torch.tensor([[32.0]], dtype=torch.float32)
    out = model(dummy_graph, bs)
    assert out.shape == (1, 2)

def test_output_is_2_dim(dummy_graph):
    model = ArchitectureGNN(node_feature_dim=12, hidden_dim=64)
    bs = torch.tensor([[32.0]], dtype=torch.float32)
    out = model(dummy_graph, bs)
    assert out.shape[1] == 2

def test_no_nan_in_output(dummy_graph):
    model = ArchitectureGNN()
    bs = torch.tensor([[16.0]], dtype=torch.float32)
    out = model(dummy_graph, bs)
    assert not torch.isnan(out).any()

def test_different_graphs_different_output(dummy_graph, another_dummy_graph):
    model = ArchitectureGNN()
    model.eval()
    bs = torch.tensor([[1.0]], dtype=torch.float32)
    with torch.no_grad():
        out1 = model(dummy_graph, bs)
        out2 = model(another_dummy_graph, bs)
    assert not torch.allclose(out1, out2)

def test_batch_size_scaling_impacts_output(dummy_graph):
    model = ArchitectureGNN()
    model.eval()
    bs1 = torch.tensor([[1.0]], dtype=torch.float32)
    bs2 = torch.tensor([[64.0]], dtype=torch.float32)
    with torch.no_grad():
        out1 = model(dummy_graph, bs1)
        out2 = model(dummy_graph, bs2)
    assert not torch.allclose(out1, out2)

def test_1d_batch_size_tensor_unsqueeze(dummy_graph):
    """Covers line 47: batch_size_tensor = batch_size_tensor.unsqueeze(1)"""
    model = ArchitectureGNN()
    model.eval()
    # Pass a 1D tensor (shape [1]) instead of 2D ([1, 1]) to trigger the unsqueeze branch
    bs_1d = torch.tensor([32.0], dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy_graph, bs_1d)
    assert out.shape == (1, 2)
