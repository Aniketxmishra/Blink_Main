import time
from dynamic_gpu_predictor import DynamicGPUPredictor
from batch_size_optimizer import BatchSizeOptimizer
from dynamic_predictor import DynamicPredictor

def test_dynamic_gpu_predictor():
    print("\nTesting Dynamic GPU Predictor...")
    start_time = time.time()
    
    # Initialize the predictor
    print("Initializing predictor...")
    predictor = DynamicGPUPredictor()
    print(f"Initialization time: {time.time() - start_time:.2f} seconds")
    
    # Create model features (only numeric features)
    model_features = {
        "total_parameters": 525354,
        "trainable_parameters": 525354,
        "model_size_mb": 2.004066467285156
    }

    # Get optimized prediction
    print("\nMaking prediction...")
    pred_start = time.time()
    result = predictor.predict_and_optimize(model_features)
    print(f"Prediction time: {time.time() - pred_start:.2f} seconds")
    
    print(f"Optimal batch size: {result['optimal_batch_size']}")
    print(f"Predicted execution time: {result['predicted_execution_time']:.2f} ms")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

def test_batch_size_optimizer():
    print("\nTesting Batch Size Optimizer...")
    start_time = time.time()
    
    print("Initializing predictor and optimizer...")
    predictor = DynamicPredictor()
    optimizer = BatchSizeOptimizer(predictor)
    print(f"Initialization time: {time.time() - start_time:.2f} seconds")

    # Test with VGG16 features (only numeric features)
    vgg16_features = {
        "total_parameters": 138357544,
        "trainable_parameters": 138357544,
        "model_size_mb": 527.7921447753906
    }

    print("\nFinding optimal batch size...")
    opt_start = time.time()
    optimal_batch = optimizer.find_optimal_batch_size(vgg16_features)
    print(f"Optimization time: {time.time() - opt_start:.2f} seconds")
    
    print(f"Optimal batch size for VGG16: {optimal_batch}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    print("Starting tests...")
    test_dynamic_gpu_predictor()
    test_batch_size_optimizer() 