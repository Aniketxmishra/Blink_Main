import torch
import time
import os
import sys
import subprocess
from pathlib import Path
import importlib.util

# Add your prediction API
from prediction_api import load_model, extract_model_features, predict_execution_time

def clone_github_repo(repo_url, target_dir):
    """Clone a GitHub repository if it doesn't exist"""
    if not os.path.exists(target_dir):
        print(f"Cloning {repo_url} to {target_dir}...")
        subprocess.check_call(['git', 'clone', repo_url, target_dir])
    else:
        print(f"Repository already exists at {target_dir}")

def load_github_model(model_path, model_class_name):
    """Dynamically load a model from a Python file"""
    # Get the module name from the file path
    module_name = Path(model_path).stem
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Get the model class
    model_class = getattr(module, model_class_name)
    
    # Instantiate the model
    model = model_class()
    
    return model

def measure_actual_execution_time(model, input_shape, batch_sizes=[1, 2, 4], num_iterations=10):
    """Measure actual execution time for a model"""
    device = torch.device("cpu")  # Use CPU for consistency with your data
    model = model.to(device)
    model.eval()
    
    results = []
    
    for batch_size in batch_sizes:
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape, device=device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        
        # Measure execution time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end_time = time.time()
        
        # Calculate average execution time
        avg_execution_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
        
        results.append({
            "batch_size": batch_size,
            "actual_execution_time_ms": avg_execution_time
        })
        
        print(f"Batch size {batch_size}: {avg_execution_time:.2f} ms")
    
    return results

def main():
    # Example: Use a simple model from GitHub
    repo_url = "https://github.com/pytorch/vision.git"  # Using torchvision as an example
    target_dir = "github_models/vision"
    
    # Clone the repository
    clone_github_repo(repo_url, target_dir)
    
    # Path to a model definition file (using a simple model from torchvision)
    model_path = f"{target_dir}/torchvision/models/resnet.py"
    
    # Load a model from the repository
    # For this example, we're loading ResNet18 from torchvision
    # In a real scenario, you might need to adjust this based on the repository structure
    try:
        # Try to load directly from torchvision if available
        import torchvision.models as models
        model = models.resnet18(weights=None)
        print("Loaded ResNet18 from torchvision")
    except:
        # Fallback to loading from the cloned repository
        model = load_github_model(model_path, "ResNet")
        # Initialize with appropriate parameters for ResNet18
        model = model(num_classes=1000, block=None, layers=[2, 2, 2, 2])
        print("Loaded ResNet18 from cloned repository")
    
    # Input shape for the model
    input_shape = (3, 224, 224)
    batch_sizes = [1, 2, 4]
    
    # Load your prediction model
    prediction_model = load_model('models/gradient_boosting_model.joblib')
    
    # Extract features from the model
    features = extract_model_features(model, input_shape)
    
    # Predict execution time
    predictions = predict_execution_time(prediction_model, features, batch_sizes)
    
    # Measure actual execution time
    actual_times = measure_actual_execution_time(model, input_shape, batch_sizes)
    
    # Compare predictions with actual measurements
    print("\nComparison of Predicted vs Actual Execution Times:")
    print("-" * 60)
    print(f"{'Batch Size':<10} {'Predicted (ms)':<15} {'Actual (ms)':<15} {'Error (%)':<10}")
    print("-" * 60)
    
    for i, batch_size in enumerate(batch_sizes):
        predicted = predictions[i]["predicted_execution_time_ms"]
        actual = actual_times[i]["actual_execution_time_ms"]
        error_percent = abs(predicted - actual) / actual * 100
        
        print(f"{batch_size:<10} {predicted:<15.2f} {actual:<15.2f} {error_percent:<10.2f}")

if __name__ == "__main__":
    main()
