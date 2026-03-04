import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import concurrent.futures
from collections import defaultdict

class ModelAnalyzer:
    """Scalable model architecture analyzer with parallel processing"""
    
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
    
    def extract_features(self, model, input_shape=(3, 224, 224)):
        """Extract features from a PyTorch model efficiently"""
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get model size in MB
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        # Count layer types in parallel
        layer_counts = self._count_layer_types(model)
        
        # Extract architecture patterns
        architecture_patterns = self._extract_architecture_patterns(model)
        
        # Combine all features
        features = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "layer_counts": layer_counts,
            "architecture_patterns": architecture_patterns
        }
        
        return features
    
    def _count_layer_types(self, model):
        """Count different types of layers in the model"""
        layer_counts = defaultdict(int)
        
        # Process in parallel for large models
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for name, module in model.named_children():
                futures.append(executor.submit(self._process_module, module))
            
            # Combine results
            for future in concurrent.futures.as_completed(futures):
                module_counts = future.result()
                for layer_type, count in module_counts.items():
                    layer_counts[layer_type] += count
        
        return dict(layer_counts)
    
    def _process_module(self, module):
        """Process a module to count layer types (for parallel execution)"""
        counts = defaultdict(int)
        module_type = module.__class__.__name__
        counts[module_type] += 1
        
        # Recursively process children
        for child in module.children():
            child_type = child.__class__.__name__
            counts[child_type] += 1
            
        return counts
    
    def _extract_architecture_patterns(self, model):
        """Extract architectural patterns like skip connections, attention, etc."""
        patterns = {
            "has_skip_connections": False,
            "has_attention": False,
            "has_normalization": False,
            "max_depth": 0
        }
        
        # Check for skip connections (simplified)
        for name, module in model.named_modules():
            if "resnet" in model.__class__.__name__.lower() or "residual" in name.lower():
                patterns["has_skip_connections"] = True
                
            # Check for attention mechanisms
            if "attention" in name.lower() or "mha" in name.lower():
                patterns["has_attention"] = True
                
            # Check for normalization
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                patterns["has_normalization"] = True
        
        # Estimate depth
        patterns["max_depth"] = self._estimate_model_depth(model)
        
        return patterns
    
    def _estimate_model_depth(self, model):
        """Estimate the depth of the model"""
        def count_layers(module, depth=1):
            max_depth = depth
            for child in module.children():
                if list(child.children()):  # If has children, recurse
                    child_depth = count_layers(child, depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    max_depth = max(max_depth, depth + 1)
            return max_depth
            
        return count_layers(model)
    
    def analyze_batch(self, models, input_shapes=None):
        """Analyze multiple models in parallel"""
        if input_shapes is None:
            input_shapes = [(3, 224, 224)] * len(models)
            
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.extract_features, model, shape): i 
                      for i, (model, shape) in enumerate(zip(models, input_shapes))}
            
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results.append((idx, future.result()))
                
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
