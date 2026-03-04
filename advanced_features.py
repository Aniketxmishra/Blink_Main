import torch
import torch.nn as nn
import numpy as np
from collections import Counter

def count_layer_types(model):
    """Count different types of layers in the model"""
    layer_types = Counter()
    for module in model.modules():
        layer_types[module.__class__.__name__] += 1
    
    # Remove the model itself from count
    layer_types[model.__class__.__name__] -= 1
    
    return dict(layer_types)

def compute_flops(model, input_shape):
    """Estimate FLOPs for a model (simplified)"""
    # This is a simplified estimation
    # For more accurate results, use libraries like thop or fvcore
    try:
        from thop import profile
        device = torch.device("cpu")
        dummy_input = torch.randn(1, *input_shape, device=device)
        macs, _ = profile(model, inputs=(dummy_input,))
        flops = macs * 2  # Approximate FLOPs as 2 * MACs
        return flops
    except:
        # Fallback to a simple heuristic if thop is not available
        total_params = sum(p.numel() for p in model.parameters())
        return total_params * 2  # Very rough approximation

def compute_memory_access_pattern(model):
    """Estimate memory access patterns"""
    # Count sequential vs. random access operations
    sequential_ops = 0
    random_ops = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # These typically have more random access patterns
            random_ops += 1
        elif isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d)):
            # These typically have more sequential access patterns
            sequential_ops += 1
    
    total_ops = sequential_ops + random_ops
    if total_ops == 0:
        return 0.5  # Default value
    
    # Return ratio of sequential to random (higher means more sequential)
    return sequential_ops / total_ops if total_ops > 0 else 0.5

def extract_advanced_features(model, input_shape):
    """Extract advanced features from a model"""
    # Basic model info
    total_params = sum(p.numel() for p in model.parameters())
    
    # Layer type counts
    layer_counts = count_layer_types(model)
    
    # Compute FLOPs
    flops = compute_flops(model, input_shape)
    
    # Memory access patterns
    memory_pattern = compute_memory_access_pattern(model)
    
    # Model depth (number of layers)
    model_depth = len(list(model.modules())) - 1  # Subtract 1 for the model itself
    
    # Compute-to-memory ratio (FLOPs per parameter)
    compute_memory_ratio = flops / total_params if total_params > 0 else 0
    
    features = {
        "total_parameters": total_params,
        "flops": flops,
        "compute_memory_ratio": compute_memory_ratio,
        "memory_access_pattern": memory_pattern,
        "model_depth": model_depth,
        "layer_counts": layer_counts
    }
    
    # Add specific layer type counts as individual features
    for layer_type, count in layer_counts.items():
        features[f"num_{layer_type}"] = count
    
    return features
