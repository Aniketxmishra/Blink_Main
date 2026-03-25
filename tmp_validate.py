from blink.gpu_predictor import GPUPredictor
import torchvision.models as models

print("1. Verifying feature count...")
predictor = GPUPredictor()
assert len(predictor.feature_cols) == 21, f"Expected 21, got {len(predictor.feature_cols)}"
print(f"Feature count: {len(predictor.feature_cols)} - OK")

print("\n2. Verifying custom model prediction (also checks XGBoost column mismatch)...")
result = predictor.predict_for_custom_model(models.resnet50(), batch_size=16)
print("Prediction Result:")
import pprint
pprint.pprint(result)

print("\n3 & 4. Verifying cache key stability...")
key1 = predictor._get_cache_key({'flops': 1e9, 'model_size_mb': 25, 'batch_size': 16})
key2 = predictor._get_cache_key({'flops': 1e9, 'model_size_mb': 25, 'batch_size': 16})
assert key1 == key2, "Cache keys do not match!"
print("Cache key stability - OK")
print(f"Sample key: {key1}")

print("\nAll verification steps OK")
