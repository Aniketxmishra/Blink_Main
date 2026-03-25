import torchvision.models as models
from gpu_predictor import GPUPredictor
p = GPUPredictor()
assert len(p.feature_cols) == 19
result = p.predict_for_custom_model(models.resnet50(), batch_size=16)
print(result)
assert 'exec_time_ms' in result
assert result['exec_time_ms'] > 0
print("OK")
