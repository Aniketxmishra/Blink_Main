from model_profiler import ModelProfiler
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
import time
import pandas as pd
from datetime import datetime

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect GPU usage data for various models')
    parser.add_argument('--batch-sizes', type=int, nargs='+', 
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                        help='Batch sizes to profile')
    parser.add_argument('--save-dir', type=str, default='data/raw',
                        help='Directory to save profiling results')
    parser.add_argument('--model-type', type=str, default='all',
                        choices=['all', 'cnn', 'transformer', 'custom'],
                        help='Type of models to profile')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of profiling runs per config (results averaged)')
    args = parser.parse_args()
    
    # Create a model profiler
    profiler = ModelProfiler(save_dir=args.save_dir)
    
    # Profile CNN models
    if args.model_type in ['all', 'cnn']:
        print("Profiling CNN models...")
        cnn_results = profile_cnn_models(profiler, args.batch_sizes)
    
    # Profile transformer models
    if args.model_type in ['all', 'transformer']:
        print("Profiling transformer models...")
        transformer_results = profile_transformer_models(profiler, args.batch_sizes)
    
    # Profile custom models
    if args.model_type in ['all', 'custom']:
        print("Profiling custom models...")
        custom_results = profile_custom_models(profiler, args.batch_sizes)
    
    print("Data collection complete!")

def profile_cnn_models(profiler, batch_sizes):
    """Profile common CNN architectures"""
    cnn_models = {
        # Original models
        "resnet18": models.resnet18(weights=None),
        "resnet50": models.resnet50(weights=None),
        "mobilenet_v2": models.mobilenet_v2(weights=None),
        "densenet121": models.densenet121(weights=None),
        "vgg16": models.vgg16(weights=None),
        "efficientnet_b0": models.efficientnet_b0(weights=None),
        "regnet_y_400mf": models.regnet_y_400mf(weights=None),
        "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0(weights=None),
        "squeezenet1_0": models.squeezenet1_0(weights=None),
        "wide_resnet50_2": models.wide_resnet50_2(weights=None),
        "convnext_tiny": models.convnext_tiny(weights=None),
        # New models for 30+ total
        "efficientnet_v2_s": models.efficientnet_v2_s(weights=None),
        "convnext_small": models.convnext_small(weights=None),
        "convnext_base": models.convnext_base(weights=None),
        "regnet_x_400mf": models.regnet_x_400mf(weights=None),
        "regnet_y_800mf": models.regnet_y_800mf(weights=None),
        "mnasnet1_0": models.mnasnet1_0(weights=None),
        "googlenet": models.googlenet(weights=None),
        "inception_v3": models.inception_v3(weights=None),
        "resnext50_32x4d": models.resnext50_32x4d(weights=None),
        "mobilenet_v3_large": models.mobilenet_v3_large(weights=None),
        "mobilenet_v3_small": models.mobilenet_v3_small(weights=None),
        "densenet169": models.densenet169(weights=None),
        "densenet201": models.densenet201(weights=None),
        "vgg19": models.vgg19(weights=None),
    }
    
    all_results = []
    
    for name, model in cnn_models.items():
        print(f"Profiling {name}...")
        results = profiler.profile_model(
            model=model,
            input_shape=(3, 224, 224),
            batch_sizes=batch_sizes,
            model_name=name
        )
        all_results.append(results)
    
    return all_results

def profile_transformer_models(profiler, batch_sizes):
    """Profile transformer models"""
    try:
        # Import transformers only if needed
        from transformers import BertModel, RobertaModel, GPT2Model
        
        transformer_models = {}
        
        # Try to load models
        try:
            transformer_models["bert-base"] = BertModel.from_pretrained("bert-base-uncased")
        except Exception as e:
            print(f"Could not load BERT: {e}")
        
        try:
            transformer_models["roberta-base"] = RobertaModel.from_pretrained("roberta-base")
        except Exception as e:
            print(f"Could not load RoBERTa: {e}")
            
        try:
            transformer_models["gpt2"] = GPT2Model.from_pretrained("gpt2")
        except Exception as e:
            print(f"Could not load GPT-2: {e}")
        
        all_results = []
        
        for name, model in transformer_models.items():
            print(f"Profiling {name}...")
            
            # Create a custom profiling function for transformers
            results = profile_transformer_model(
                profiler=profiler,
                model=model,
                batch_sizes=batch_sizes,
                model_name=name
            )
            all_results.append(results)
        
        return all_results
    
    except ImportError:
        print("Transformers library not installed. Skipping transformer models.")
        return []

def profile_transformer_model(profiler, model, batch_sizes, model_name):
    """Custom profiling for transformer models with appropriate input types"""
    results = []
    
    device = torch.device("cuda" if profiler.cuda_available else "cpu")
    model = model.to(device)
    model.eval()
    
    # Calculate model parameters and size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model size in MB
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    for batch_size in batch_sizes:
        try:
            # Create appropriate input tensors for transformers
            # Use random integers in a valid range for token IDs (e.g., 0-30000)
            input_ids = torch.randint(0, 30000, (batch_size, 128), device=device)
            attention_mask = torch.ones((batch_size, 128), device=device)
            
            # Measure execution time and peak memory
            timing_result = profiler._profile_execution_time(
                model, 
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            
            # Get GPU metrics if available
            gpu_metrics = profiler.get_gpu_utilization() if profiler.cuda_available else {"error": "CUDA not available"}
            
            # Record results
            result = {
                "model_name": model_name,
                "batch_size": batch_size,
                "input_shape": "(128,)",
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": model_size_mb,
                "device": str(device)
            }
            
            result.update(timing_result)
            
            # Add GPU metrics if available
            if isinstance(gpu_metrics, dict) and "error" not in gpu_metrics:
                result.update(gpu_metrics)
            
            results.append(result)
            
            # Print progress
            print(f"Profiled {model_name} with batch size {batch_size}: {timing_result['execution_time_ms']:.2f} ms")
            
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at batch size {batch_size} for {model_name}. Skipping larger batches.")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"Error at batch size {batch_size} for {model_name}: {e}")
            break
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{profiler.save_dir}/{model_name}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    return df

def profile_custom_models(profiler, batch_sizes):
    """Profile custom model architectures"""
    
    class SimpleCNN(nn.Module):
        def __init__(self, num_layers=3, channels=16):
            super(SimpleCNN, self).__init__()
            layers = []
            in_channels = 3
            
            for i in range(num_layers):
                out_channels = channels * (2 ** i)
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2))
                in_channels = out_channels
            
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(out_channels * (224 // (2**num_layers)) * (224 // (2**num_layers)), 10)
            
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    custom_models = {
        "simple_cnn_3layers": SimpleCNN(num_layers=3, channels=16),
        "simple_cnn_5layers": SimpleCNN(num_layers=5, channels=16),
        "simple_cnn_3layers_wide": SimpleCNN(num_layers=3, channels=32),
    }
    
    all_results = []
    
    for name, model in custom_models.items():
        print(f"Profiling {name}...")
        results = profiler.profile_model(
            model=model,
            input_shape=(3, 224, 224),
            batch_sizes=batch_sizes,
            model_name=name
        )
        all_results.append(results)
    
    return all_results

if __name__ == "__main__":
    main()
