"""
blink._predictor
================
BlinkPredictor — public-facing facade over gpu_predictor.GPUPredictor.
"""
from __future__ import annotations

from pathlib import Path
import torch.nn as nn

_DEFAULT_MODELS_DIR = Path(__file__).parent / "weights"


class BlinkPredictor:
    """
    Predict GPU execution time and peak memory for a PyTorch model.

    Examples
    --------
    >>> from blink import BlinkPredictor
    >>> p = BlinkPredictor()
    >>> result = p.predict("resnet50", batch_size=16)
    >>> result["exec_time_ms"]
    28.4
    """

    NAMED_MODELS = {
        "resnet18", "resnet50", "vgg16", "mobilenet_v2", "densenet121",
    }

    def __init__(self, models_dir: str | Path | None = None):
        self._models_dir = Path(models_dir) if models_dir else _DEFAULT_MODELS_DIR
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            from .gpu_predictor import GPUPredictor          # ✅ relative
            self._predictor = GPUPredictor(
                model_path=str(self._models_dir / "median_quantile_(0.5)_model.joblib"),
                memory_model_path=str(self._models_dir / "memory_model.joblib"),
            )
        return self._predictor

    def predict(
        self,
        model: str | nn.Module,
        batch_size: int = 32,
        input_shape: tuple = (3, 224, 224),
    ) -> dict:
        predictor = self._get_predictor()

        if isinstance(model, str):
            model = _load_named_model(model)

        from .model_analyser import ModelAnalyzer            # ✅ relative
        analyser = ModelAnalyzer()
        features = analyser.extract_features(model, input_shape)
        features["batch_size"] = batch_size

        raw = predictor.predict([features])
        if isinstance(raw, list):
            raw = raw[0]

        return {
            "exec_time_ms":    raw.get("exec_time_ms", 0.0),
            "exec_time_lower": raw.get("exec_time_lower", 0.0),
            "exec_time_upper": raw.get("exec_time_upper", 0.0),
            "memory_mb":       raw.get("memory_usage_mb", 0.0),
            "memory_lower_mb": raw.get("memory_lower_mb", 0.0),
            "memory_upper_mb": raw.get("memory_upper_mb", 0.0),
            "source":          raw.get("source", "blink"),
        }

    def predict_batch(
        self,
        model: str | nn.Module,
        batch_sizes: list[int],
        input_shape: tuple = (3, 224, 224),
    ) -> list[dict]:
        return [self.predict(model, bs, input_shape) for bs in batch_sizes]


def _load_named_model(name: str) -> nn.Module:
    import torchvision.models as tv
    name = name.lower().replace("-", "_")
    loaders = {
        "resnet18":     lambda: tv.resnet18(weights=None),
        "resnet50":     lambda: tv.resnet50(weights=None),
        "vgg16":        lambda: tv.vgg16(weights=None),
        "mobilenet_v2": lambda: tv.mobilenet_v2(weights=None),
        "densenet121":  lambda: tv.densenet121(weights=None),
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Supported names: {sorted(loaders)}. "
            "Or pass an nn.Module directly."
        )
    return loaders[name]()
