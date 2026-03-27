import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from blink.feature_extractor import ModelFeatureExtractor


def load_model(model_path='models/random_forest_model.joblib', memory_model_path='models/memory_model.joblib'):
    """Load the trained prediction models"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Loaded execution time prediction model from {model_path}")
    
    # Load interval models for execution time
    try:
        model_lower = joblib.load('models/execution_lower_model.joblib')
        model_upper = joblib.load('models/execution_upper_model.joblib')
    except FileNotFoundError:
        model_lower = model_upper = None
    
    memory_model = None
    memory_lower = memory_upper = None
    if os.path.exists(memory_model_path):
        memory_model = joblib.load(memory_model_path)
        print(f"Loaded memory prediction model from {memory_model_path}")
        try:
           memory_lower = joblib.load('models/memory_lower_model.joblib')
           memory_upper = joblib.load('models/memory_upper_model.joblib')
        except FileNotFoundError:
           memory_lower = memory_upper = None
    else:
        print(f"Warning: Memory model not found at {memory_model_path}")
        
    bounds = {
        'exec_lower': model_lower,
        'exec_upper': model_upper,
        'mem_lower': memory_lower,
        'mem_upper': memory_upper
    }
        
    return model, memory_model, bounds

def extract_model_features(model, input_shape):
    """Extract features from a PyTorch model"""
    # Use the same feature extractor as training
    extractor = ModelFeatureExtractor(save_dir='data/temp')
    # Use CPU for extraction by default as in feature_extractor
    device = next(model.parameters()).device
    model.to("cpu")
    features = extractor.extract_model_features(model, input_shape)
    model.to(device)
    
    return features

def predict_execution_time(prediction_model, bounds, model_features, batch_sizes=[1, 2, 4, 8]):
    """Predict execution time for different batch sizes"""
    predictions = []
    
    for batch_size in batch_sizes:
        # Create feature vector for prediction
        # Convert to DataFrame for prediction
        # Map values to the exact feature columns expected by the model
        feature_cols = [
            'batch_size',
            'flops',
            'compute_memory_ratio',
            'num_conv_layers',
            'num_fc_layers',
            'num_bn_layers',
            'avg_conv_kernel_size',
            'max_conv_channels',
            'total_conv_params',
            'total_fc_params',
            'model_depth',
            'model_size_mb'
        ]
        
        row = {}
        for col in feature_cols:
            if col == 'batch_size':
                row[col] = batch_size
            else:
                row[col] = model_features.get(col, 0)
            
        X = pd.DataFrame([row])
        
        # Predict execution time (log scale -> standard scale)
        execution_time_log = prediction_model.predict(X)[0]
        # Clip log-space to sane range: e^(-2)=0.14ms to e^(10)=22sec max
        execution_time_log = np.clip(execution_time_log, -2, 10)
        execution_time = np.expm1(execution_time_log)
        
        # Add bounds
        lower_bound = execution_time
        upper_bound = execution_time
        if bounds and bounds.get('exec_lower') and bounds.get('exec_upper'):
            lower_bound = np.expm1(bounds['exec_lower'].predict(X)[0])
            upper_bound = np.expm1(bounds['exec_upper'].predict(X)[0])
            # Ensure logical consistency
            lower_bound = max(1.0, lower_bound)
            upper_bound = max(execution_time, upper_bound)
        
        predictions.append({
            "batch_size": batch_size,
            "predicted_execution_time_ms": execution_time,
            "exec_lower_ms": lower_bound,
            "exec_upper_ms": upper_bound
        })
    
    return predictions

def predict_memory_usage(memory_model, bounds, model_features, batch_sizes=[1, 2, 4, 8]):
    """Predict memory usage for different batch sizes"""
    predictions = []
    
    feature_cols = [
        'batch_size',
        'flops',
        'compute_memory_ratio',
        'num_conv_layers',
        'num_fc_layers',
        'num_bn_layers',
        'avg_conv_kernel_size',
        'max_conv_channels',
        'total_conv_params',
        'total_fc_params',
        'model_depth',
        'model_size_mb'
    ]
    
    for batch_size in batch_sizes:
        if memory_model is not None:
            row = {}
            for col in feature_cols:
                row[col] = batch_size if col == 'batch_size' else model_features.get(col, 0)
            X = pd.DataFrame([row])
            estimated_memory = memory_model.predict(X)[0]
            
            # Predict bounds
            if bounds and bounds.get('mem_lower') and bounds.get('mem_upper'):
                mem_lower = bounds['mem_lower'].predict(X)[0]
                mem_upper = bounds['mem_upper'].predict(X)[0]
                # Consistency adjustments
                mem_lower = max(10.0, mem_lower)
                mem_upper = max(estimated_memory, mem_upper)
            else:
                mem_lower = mem_upper = estimated_memory
                
        else:
            base_memory = model_features.get("model_size_mb", 0)
            estimated_memory = base_memory * (1 + 0.2 * (batch_size - 1))
            mem_lower = estimated_memory * 0.9
            mem_upper = estimated_memory * 1.1
            
        predictions.append({
            "batch_size": batch_size,
            "predicted_memory_usage_mb": estimated_memory,
            "memory_lower_mb": mem_lower,
            "memory_upper_mb": mem_upper
        })
    
    return predictions

def predict_with_gnn(model: nn.Module, batch_size: int):
    """Predict execution time and memory usage for a custom PyTorch model using GNN or fallback to XGBoost"""
    # Import here to avoid circular dependencies
    from blink.gpu_predictor import GPUPredictor
    
    predictor = GPUPredictor()
    result = predictor.predict_for_custom_model(model, batch_size)
    
    return {
        "exec_time_ms": result['exec_time_ms'],
        "memory_mb": result['memory_usage_mb'],
        "confidence_lower": result['exec_lower_ms'],
        "confidence_upper": result['exec_upper_ms'],
        "source": result.get('source', 'unknown')
    }

def predict_for_custom_model(prediction_model, memory_model, bounds, custom_model, input_shape, batch_sizes=[1, 2, 4, 8]):
    """Predict execution time and memory usage for a custom PyTorch model"""
    # Extract features from the model
    features = extract_model_features(custom_model, input_shape)
    
    # Predict execution time
    time_predictions = predict_execution_time(prediction_model, bounds, features, batch_sizes)
    
    # Predict memory usage
    memory_predictions = predict_memory_usage(memory_model, bounds, features, batch_sizes)
    
    # Combine predictions
    combined_predictions = []
    for i in range(len(batch_sizes)):
        pred_time = time_predictions[i]
        pred_mem = memory_predictions[i]
        
        combined_predictions.append({
            "batch_size": batch_sizes[i],
            "predicted_execution_time_ms": pred_time["predicted_execution_time_ms"],
            "exec_lower_ms": pred_time["exec_lower_ms"],
            "exec_upper_ms": pred_time["exec_upper_ms"],
            "predicted_memory_usage_mb": pred_mem["predicted_memory_usage_mb"],
            "memory_lower_mb": pred_mem["memory_lower_mb"],
            "memory_upper_mb": pred_mem["memory_upper_mb"]
        })
    
    # Print results
    print("\nModel Features:")
    print(f"  Total Parameters: {features['total_parameters']:,}")
    print(f"  Model Size: {features['model_size_mb']:.2f} MB")
    
    print("\nPredicted Performance:")
    for pred in combined_predictions:
        print(f"  Batch Size {pred['batch_size']}:")
        print(f"    Execution Time: {pred['predicted_execution_time_ms']:.2f} ms [{pred['exec_lower_ms']:.2f} - {pred['exec_upper_ms']:.2f}]")
        print(f"    Memory Usage: {pred['predicted_memory_usage_mb']:.2f} MB [{pred['memory_lower_mb']:.2f} - {pred['memory_upper_mb']:.2f}]")
    
    return combined_predictions

def create_sample_model(num_layers=3, channels=16):
    """Create a sample CNN model for testing"""
    class SampleCNN(nn.Module):
        def __init__(self, num_layers=3, channels=16):
            super().__init__()
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
    
    return SampleCNN(num_layers, channels)

def main():
    parser = argparse.ArgumentParser(description='Predict GPU usage for deep learning models')
    parser.add_argument('--model-path', type=str, default='models/xgboost_(tuned)_model.joblib',
                        help='Path to the trained prediction model')
    parser.add_argument('--test-model', action='store_true',
                        help='Test with a sample model')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of layers for the test model')
    parser.add_argument('--channels', type=int, default=16,
                        help='Base number of channels for the test model')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Batch sizes to predict for')
    
    args = parser.parse_args()
    
    # Load the prediction models
    prediction_model, memory_model, bounds = load_model(args.model_path)
    
    if args.test_model:
        # Create a sample model
        print(f"Creating a sample CNN with {args.layers} layers and {args.channels} base channels...")
        sample_model = create_sample_model(args.layers, args.channels)
        
        # Predict execution time
        predict_for_custom_model(
            prediction_model, 
            memory_model,
            bounds,
            sample_model, 
            (3, 224, 224), 
            args.batch_sizes
        )
    else:
        print("Use --test-model to test with a sample model")
        print("You can also import this module and use predict_for_custom_model() with your own models")

if __name__ == "__main__":
    main()
