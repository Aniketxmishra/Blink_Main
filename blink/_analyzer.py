"""
blink._analyzer
===============
BlinkAnalyzer — public-facing facade over ModelAnalyzer.
Extracts architecture features from any PyTorch nn.Module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class BlinkAnalyzer:
    """
    Extract architecture features from a PyTorch model.

    Used internally by BlinkPredictor, but also useful standalone
    if you want to inspect what Blink sees about your model.

    Examples
    --------
    >>> from blink import BlinkAnalyzer
    >>> import torchvision.models as tv
    >>> model = tv.resnet18(weights=None)
    >>> feats = BlinkAnalyzer().analyze(model)
    >>> feats["flops"]
    1814073344
    >>> feats["num_conv_layers"]
    20
    """

    def __init__(self):
        self._analyser = None

    def _get(self):
        if self._analyser is None:
            from .model_analyser import ModelAnalyzer
            self._analyser = ModelAnalyzer()
        return self._analyser

    def analyze(
        self,
        model: nn.Module,
        input_shape: tuple = (3, 224, 224),
    ) -> dict:
        """
        Extract architecture features from *model*.

        Parameters
        ----------
        model : nn.Module
            Any PyTorch model.
        input_shape : tuple
            (C, H, W) input shape. Default ``(3, 224, 224)``.

        Returns
        -------
        dict
            Architecture features used by Blink's prediction models, e.g.
            ``flops``, ``num_conv_layers``, ``model_size_mb``, etc.
        """
        return self._get().extract_features(model, input_shape)

    def summary(self, model: nn.Module, input_shape: tuple = (3, 224, 224)) -> str:
        """
        Return a human-readable string summary of the model's key metrics.

        Examples
        --------
        >>> print(BlinkAnalyzer().summary(model))
        Model Architecture Summary
        ==========================
        Parameters  :  11,689,512
        FLOPs       :   1,814 M
        Conv layers :        20
        Size (MB)   :   44.59
        """
        feats = self.analyze(model, input_shape)
        flops_m = feats.get('flops', 0) / 1e6
        lines = [
            "Model Architecture Summary",
            "==========================",
            f"Parameters  : {feats.get('total_parameters', 0):>12,}",
            f"FLOPs       : {flops_m:>10.1f} M",
            f"Conv layers : {feats.get('num_conv_layers', 0):>10}",
            f"FC layers   : {feats.get('num_fc_layers', 0):>10}",
            f"BN layers   : {feats.get('num_bn_layers', 0):>10}",
            f"Model depth : {feats.get('model_depth', 0):>10}",
            f"Size (MB)   : {feats.get('model_size_mb', 0):>10.2f}",
        ]
        return "\n".join(lines)
