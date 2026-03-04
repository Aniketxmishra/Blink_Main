import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import json
from datetime import datetime

class ModelFeatureExtractor:
    """Extract features from PyTorch models for GPU usage prediction"""
    
    def __init__(self, save_dir='data/processed'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def count_ops_and_params(self, model, input_shape):
        """Count operations (FLOPs) and parameters for a model"""
        from thop import profile
        
        device = torch.device("cpu")  # Use CPU for counting
        model = model.to(device)
        dummy_input = torch.randn(1, *input_shape, device=device)
        
        try:
            macs, params = profile(model, inputs=(dummy_input,))
            flops = macs * 2  # Multiply MACs by 2 to get FLOPs
            return flops, params
        except Exception as e:
            print(f"Error in counting ops: {e}")
            return 0, sum(p.numel() for p in model.parameters())
    
    def extract_layer_info(self, model):
        """Extract information about each layer in the model"""
        layer_info = []
        
        for name, module in model.named_modules():
            if name == '':  # Skip the model itself
                continue
                
            # Get layer type
            layer_type = module.__class__.__name__
            
            # Get layer parameters
            params = sum(p.numel() for p in module.parameters())
            
            # Get layer-specific attributes
            attrs = {}
            
            # Convolutional layers
            if isinstance(module, nn.Conv2d):
                attrs = {
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                    'stride': module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                    'padding': module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                    'groups': module.groups
                }
            
            # Linear layers
            elif isinstance(module, nn.Linear):
                attrs = {
                    'in_features': module.in_features,
                    'out_features': module.out_features
                }
            
            # Pooling layers
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                attrs = {
                    'kernel_size': module.kernel_size if hasattr(module, 'kernel_size') else 0,
                    'stride': module.stride if hasattr(module, 'stride') else 0,
                    'padding': module.padding if hasattr(module, 'padding') else 0
                }
            
            # Normalization layers
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if isinstance(module, nn.BatchNorm2d):
                    attrs = {
                        'num_features': module.num_features
                    }
                else:  # LayerNorm
                    attrs = {
                        'normalized_shape': module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
                    }
            
            # Add layer to info list
            layer_info.append({
                'name': name,
                'type': layer_type,
                'params': params,
                'attributes': attrs
            })
        
        return layer_info
    
    def extract_model_features(self, model, input_shape, model_name="unknown"):
        """Extract comprehensive features from a model"""
        # Get basic model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count operations
        flops, _ = self.count_ops_and_params(model, input_shape)
        
        # Get model size in MB
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        # Count layer types
        layer_counts = defaultdict(int)
        for name, module in model.named_modules():
            if name == '':  # Skip the model itself
                continue
            layer_counts[module.__class__.__name__] += 1
        
        # Extract detailed layer info
        layer_info = self.extract_layer_info(model)
        
        # Calculate derived features
        conv_layers = [layer for layer in layer_info if layer['type'] == 'Conv2d']
        linear_layers = [layer for layer in layer_info if layer['type'] == 'Linear']
        
        avg_conv_kernel_size = np.mean([layer['attributes']['kernel_size'] for layer in conv_layers]) if conv_layers else 0
        max_conv_channels = max([layer['attributes']['out_channels'] for layer in conv_layers]) if conv_layers else 0
        total_conv_params = sum([layer['params'] for layer in conv_layers])
        
        max_fc_size = max([layer['attributes']['out_features'] for layer in linear_layers]) if linear_layers else 0
        total_fc_params = sum([layer['params'] for layer in linear_layers])
        
        # Compute memory access patterns (simplified)
        memory_read_write_ratio = 0.5  # Simplified assumption
        
        # Compute compute-to-memory ratio (FLOPs per byte)
        # Assuming 4 bytes per parameter for forward pass
        memory_bytes = total_params * 4
        compute_memory_ratio = flops / memory_bytes if memory_bytes > 0 else 0
        
        # Create feature dictionary
        features = {
            "model_name": model_name,
            "input_shape": str(input_shape),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "flops": flops,
            "compute_memory_ratio": compute_memory_ratio,
            "memory_read_write_ratio": memory_read_write_ratio,
            "layer_counts": dict(layer_counts),
            "num_conv_layers": layer_counts.get("Conv2d", 0),
            "num_fc_layers": layer_counts.get("Linear", 0),
            "num_bn_layers": layer_counts.get("BatchNorm2d", 0),
            "avg_conv_kernel_size": avg_conv_kernel_size,
            "max_conv_channels": max_conv_channels,
            "total_conv_params": total_conv_params,
            "max_fc_size": max_fc_size,
            "total_fc_params": total_fc_params,
            "model_depth": len(layer_info)
        }
        
        return features
    
    def process_profiling_data(self, profile_data_path, output_filename=None):
        """Process profiling data from a CSV file"""
        try:
            df = pd.read_csv(profile_data_path)
            features = []

            for _, row in df.iterrows():
                # Convert int64 to int for JSON serialization
                row_dict = {k: int(v) if isinstance(v, np.int64) else v for k, v in row.items()}
                features.append(row_dict)

            if output_filename is None:
                output_filename = f"{self.save_dir}/features_{os.path.basename(profile_data_path)}"

            # Save features as CSV
            pd.DataFrame(features).to_csv(output_filename, index=False)
            print(f"Features saved to {output_filename}")
            return features
        except Exception as e:
            print(f"Error processing {profile_data_path}: {e}")
            return None
    
    def process_all_profiling_data(self, raw_data_dir='data/raw'):
        """Process all profiling data in the specified directory"""
        if not os.path.exists(raw_data_dir):
            print(f"Directory {raw_data_dir} does not exist.")
            return []

        csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {raw_data_dir}.")
            return []

        all_features = []
        for csv_file in csv_files:
            profile_data_path = os.path.join(raw_data_dir, csv_file)
            try:
                features = self.process_profiling_data(profile_data_path)
                if features:
                    all_features.extend(features)
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

        print(f"Processed features for {len(all_features)} models")
        return all_features

# Example usage
if __name__ == "__main__":
    # Install thop for FLOPs counting if not already installed
    try:
        import thop
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "thop"])
    
    extractor = ModelFeatureExtractor(save_dir='data/processed')
    
    # Process all profiling data
    features = extractor.process_all_profiling_data()
    
    print(f"Processed features for {len(features)} models")
