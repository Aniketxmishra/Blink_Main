import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

# Define known layer types for one-hot encoding
LAYER_TYPES = ['Conv', 'Linear', 'Norm', 'Pool', 'Attention', 'Other']

def encode_layer(module):
    """Encodes a single nn.Module leaf into a fixed-length node feature vector."""
    layer_type = module.__class__.__name__
    
    # 1. One-hot encode layer type (6 dims)
    type_idx = 5 # Default to 'Other'
    if 'Conv' in layer_type:
        type_idx = 0
    elif 'Linear' in layer_type:
        type_idx = 1
    elif 'Norm' in layer_type:
        type_idx = 2
    elif 'Pool' in layer_type:
        type_idx = 3
    elif 'Attention' in layer_type:
        type_idx = 4
        
    one_hot = [0.0] * len(LAYER_TYPES)
    one_hot[type_idx] = 1.0
    
    # 2. Extract parameters count (log scale)
    params = sum(p.numel() for p in module.parameters())
    log_params = float(np.log1p(params))
    
    # 3. Extract dimensions and attributes
    in_channels = 0.0
    out_channels = 0.0
    kernel_size = 0.0
    stride = 0.0
    padding = 0.0
    
    if isinstance(module, (nn.Conv2d, nn.Conv1d)):
        in_channels = float(module.in_channels)
        out_channels = float(module.out_channels)
        kernel_size = float(module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size)
        stride = float(module.stride[0] if isinstance(module.stride, tuple) else module.stride)
        padding = float(module.padding[0] if isinstance(module.padding, tuple) else module.padding)
    elif isinstance(module, nn.Linear):
        in_channels = float(module.in_features)
        out_channels = float(module.out_features)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
        if hasattr(module, 'num_features'):
            in_channels = float(module.num_features)
            out_channels = float(module.num_features)
        elif hasattr(module, 'normalized_shape'):
            shape = module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
            in_channels = float(shape)
            out_channels = float(shape)
        elif hasattr(module, 'num_channels'):
            in_channels = float(module.num_channels)
            out_channels = float(module.num_channels)
    elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.MaxPool1d, nn.AvgPool1d)):
        if hasattr(module, 'kernel_size'):
            kernel_size = float(module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0])
        if hasattr(module, 'stride'):
            stride = float(module.stride if isinstance(module.stride, int) else module.stride[0])
        if hasattr(module, 'padding'):
            padding = float(module.padding if isinstance(module.padding, int) else module.padding[0])
    elif isinstance(module, nn.MultiheadAttention):
        in_channels = float(module.embed_dim)
        out_channels = float(module.embed_dim)
        kernel_size = float(module.num_heads) # represent num_heads conceptually as kernel size
    
    # Combine all features into a fixed length list: size = 6 + 6 = 12
    features = one_hot + [
        log_params,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding
    ]
    
    return features

def model_to_graph(model):
    """
    Iterates leaf modules, builds torch_geometric.data.Data object with sequential edges.
    """
    # Get only leaf modules (modules with no children)
    leaf_modules = [m for m in model.modules() if len(list(m.children())) == 0]
    
    nodes = []
    edges = []
    
    for i, module in enumerate(leaf_modules):
        node_feat = encode_layer(module)
        nodes.append(node_feat)
        
        if i > 0:
            # Sequential connection
            edges.append([i - 1, i])
            
    if len(nodes) == 0:
        # Fallback if empty model
        nodes = [[0.0] * 12]
        
    x = torch.tensor(nodes, dtype=torch.float32)
    
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return Data(x=x, edge_index=edge_index)

if __name__ == '__main__':
    from torchvision.models import resnet18, vgg16
    
    # Test ResNet18
    model_res = resnet18()
    data_res = model_to_graph(model_res)
    print("ResNet18 Graph Info:")
    print("x.shape:", data_res.x.shape)
    print("edge_index.shape:", data_res.edge_index.shape)
    print("Expected edges:", data_res.x.shape[0] - 1)
    
    # Test VGG16 (as requested in verification)
    model_vgg = vgg16()
    data_vgg = model_to_graph(model_vgg)
    print("\nVGG16 Graph Info:")
    print("x.shape:", data_vgg.x.shape)
    print("edge_index.shape:", data_vgg.edge_index.shape)
    print("Expected edges:", data_vgg.x.shape[0] - 1)
