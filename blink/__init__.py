"""
Blink — GPU Performance Predictor
==================================
Predict GPU execution time and memory usage for PyTorch models
*before* running them on GPU hardware.

Quick start
-----------
>>> from blink import BlinkPredictor
>>> predictor = BlinkPredictor()
>>> result = predictor.predict("resnet18", batch_size=32)
>>> print(f"Exec time: {result['exec_time_ms']:.1f} ms")
>>> print(f"Memory   : {result['memory_mb']:.1f} MB")

Or with your own model:
>>> import torch.nn as nn
>>> model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))
>>> result = BlinkPredictor().predict(model, batch_size=64)
"""
from __future__ import annotations

from blink._analyzer import BlinkAnalyzer
from blink._predictor import BlinkPredictor
from blink._version import __version__

# Create a default global instance for the shorthand API
_default_predictor = None

def predict(model, *, batch_size: int = 1, **kwargs):
    """
    Shorthand for BlinkPredictor().predict(model).
    """
    global _default_predictor
    if _default_predictor is None:
         _default_predictor = BlinkPredictor()
    return _default_predictor.predict(model, batch_size=batch_size, **kwargs)

__all__ = [
    "BlinkPredictor",
    "BlinkAnalyzer",
    "predict",
    "__version__",
]
