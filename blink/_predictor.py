"""
blink._predictor
================
BlinkPredictor — public-facing facade over gpu_predictor.GPUPredictor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

# Ensure the repo root is on sys.path so the original modules are importable,
# regardless of how/where the package was installed.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_MODELS_DIR = _ROOT / "models"


class BlinkPredictor:
    """
    Predict GPU execution time and peak memory for a PyTorch model.

    Parameters
    ----------
    models_dir : str or Path, optional
        Path to the directory containing Blink's trained model files.
        Defaults to ``<repo_root>/models``.

    Examples
    --------
    >>> from blink import BlinkPredictor
    >>> p = BlinkPredictor()

    Predict with a pre-trained model name:
    >>> result = p.predict("resnet50", batch_size=16)
    >>> result["exec_time_ms"]
    28.4

    Predict with your own nn.Module:
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
    >>> result = p.predict(model, batch_size=32)
    """

    # Supported named model aliases
    NAMED_MODELS = {
        "resnet18", "resnet50", "vgg16", "mobilenet_v2", "densenet121",
    }

    def __init__(self, models_dir: str | Path | None = None):
        self._models_dir = Path(models_dir) if models_dir else _DEFAULT_MODELS_DIR
        self._predictor = None   # lazy-loaded

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _get_predictor(self):
        if self._predictor is None:
            from gpu_predictor import GPUPredictor
            self._predictor = GPUPredictor(
                model_path=str(self._models_dir / "random_forest_model.joblib"),
                memory_model_path=str(self._models_dir / "memory_model.joblib"),
            )
        return self._predictor

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        model: str | nn.Module,
        batch_size: int = 32,
        input_shape: tuple = (3, 224, 224),
    ) -> dict:
        """
        Predict execution time and peak memory for *model* at *batch_size*.

        Parameters
        ----------
        model : str or nn.Module
            Either a model name string (e.g. ``"resnet18"``), or an
            instantiated ``torch.nn.Module``.
        batch_size : int
            Batch size to predict for.
        input_shape : tuple
            Input tensor shape (C, H, W). Default ``(3, 224, 224)``.

        Returns
        -------
        dict with keys:
            exec_time_ms       : float  — predicted execution time (ms)
            exec_time_lower    : float  — 80 % CI lower bound (ms)
            exec_time_upper    : float  — 80 % CI upper bound (ms)
            memory_mb          : float  — predicted peak memory (MB)
            memory_lower_mb    : float  — 80 % CI lower bound (MB)
            memory_upper_mb    : float  — 80 % CI upper bound (MB)
            source             : str    — which backend produced this prediction
        """
        predictor = self._get_predictor()

        # Resolve named model
        if isinstance(model, str):
            model = _load_named_model(model)

        # Extract features
        from model_analyser import ModelAnalyzer
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
        """
        Predict for multiple batch sizes in one call.

        Returns
        -------
        list[dict]
            One result dict per entry in *batch_sizes*.
        """
        return [self.predict(model, bs, input_shape) for bs in batch_sizes]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_named_model(name: str) -> nn.Module:
    """Load a torchvision pre-trained model by name."""
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
