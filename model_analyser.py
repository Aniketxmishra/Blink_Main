"""
model_analyser.py  —  Activation-Aware Model Feature Extractor

Updated (Priority 1) to compute activation_memory_mb, weight_memory_mb, and
related features at inference time so predictions are consistent with the
retrained memory model.

Activation memory estimate method:
  For each Conv2d layer we compute the output feature map size:
    H_out × W_out × C_out × batch_size × 4 bytes (float32)
  We accumulate across layers and sum as the total activation estimate.
  For Transformer/MHA layers we use a sequence-length proxy.
  This is physically grounded and does not require an actual GPU.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import concurrent.futures
from collections import defaultdict


class ModelAnalyzer:
    """Scalable model architecture analyzer with activation-aware feature extraction."""

    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    # ── Public API ───────────────────────────────────────────────────────────

    def extract_features(self, model, input_shape=(3, 224, 224), batch_size=1):
        """Extract all features needed by the prediction and memory models.

        Args:
            model: PyTorch nn.Module
            input_shape: (C, H, W) — single-sample input dimensions
            batch_size: used to scale activation memory estimates
        """
        # ── Basic params ──────────────────────────────────────────────────────
        total_params      = sum(p.numel() for p in model.parameters())
        trainable_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)

        param_size   = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size  = sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)

        # Weight memory = model weights loaded on GPU (same as model_size_mb for float32)
        weight_memory_mb = model_size_mb

        # ── Layer counting ────────────────────────────────────────────────────
        layer_counts = self._count_layer_types(model)

        # ── Detailed layer stats ──────────────────────────────────────────────
        conv_layers, linear_layers, attn_layers = self._enumerate_layers(model)

        num_conv_layers   = len(conv_layers)
        num_fc_layers     = len(linear_layers)
        num_bn_layers     = layer_counts.get('BatchNorm2d', 0) + layer_counts.get('LayerNorm', 0)
        num_attn_layers   = len(attn_layers)

        avg_conv_kernel_size = (
            float(np.mean([l['kernel'] for l in conv_layers])) if conv_layers else 0.0
        )
        max_conv_channels = max([l['out_ch'] for l in conv_layers], default=0)
        total_conv_params = sum(l['params'] for l in conv_layers)
        max_fc_size       = max([l['out_features'] for l in linear_layers], default=0)
        total_fc_params   = sum(l['params'] for l in linear_layers)

        # ── FLOPs (fast manual estimate; avoids thop dependency overhead) ─────
        flops = self._estimate_flops(model, input_shape)

        # ── Activation memory estimate ────────────────────────────────────────
        # Physically: activation memory = sum over all layers of output tensor size
        activation_memory_mb = self._estimate_activation_memory_mb(
            conv_layers, linear_layers, attn_layers, input_shape, batch_size
        )
        activation_memory_per_sample = activation_memory_mb / max(batch_size, 1)

        # Derived features
        compute_memory_ratio    = flops / max(total_params * 4, 1)
        flops_per_activation_mb = flops / max(activation_memory_mb, 1e-3)
        input_resolution_factor = int(input_shape[-2]) * int(input_shape[-1]) if len(input_shape) >= 2 else 224*224

        # ── Architecture patterns ─────────────────────────────────────────────
        architecture_patterns = self._extract_architecture_patterns(model)
        model_depth           = architecture_patterns['max_depth']

        return {
            # Basic
            'total_parameters':        total_params,
            'trainable_parameters':    trainable_params,
            'model_size_mb':           model_size_mb,

            # Activation-aware (NEW — matched to retrained memory model)
            'activation_memory_mb':         activation_memory_mb,
            'weight_memory_mb':             weight_memory_mb,
            'activation_memory_per_sample': activation_memory_per_sample,
            'flops_per_activation_mb':      flops_per_activation_mb,
            'input_resolution_factor':      input_resolution_factor,

            # Compute
            'flops':                   flops,
            'compute_memory_ratio':    compute_memory_ratio,
            'memory_read_write_ratio': 0.5,   # placeholder, not critical

            # Layer structure
            'num_conv_layers':     num_conv_layers,
            'num_fc_layers':       num_fc_layers,
            'num_bn_layers':       num_bn_layers,
            'num_attn_layers':     num_attn_layers,
            'avg_conv_kernel_size':avg_conv_kernel_size,
            'max_conv_channels':   max_conv_channels,
            'total_conv_params':   total_conv_params,
            'max_fc_size':         max_fc_size,
            'total_fc_params':     total_fc_params,
            'model_depth':         model_depth,

            # Patterns (used by execution-time predictor)
            'layer_counts':           layer_counts,
            'architecture_patterns':  architecture_patterns,
        }

    # ── Activation memory estimation ─────────────────────────────────────────

    def _estimate_activation_memory_mb(self, conv_layers, linear_layers,
                                       attn_layers, input_shape, batch_size):
        """Estimate activation memory from layer structure without running GPU.

        Formula:  activation_MB = Σ_layers (output_elements × 4 bytes) × batch_size / 1MB

        For Conv2d: output_elements = C_out × H_out × W_out
        For Linear: output_elements = out_features
        For Attention: output_elements = seq_len × hidden (proxy via attn layer size)
        """
        bytes_per_element = 4  # float32
        total_bytes = 0

        # Estimate spatial dimensions through the network (simplified tracking)
        H, W = (input_shape[-2], input_shape[-1]) if len(input_shape) >= 2 else (224, 224)

        for layer in conv_layers:
            stride = layer.get('stride', 1) or 1
            kernel = layer.get('kernel', 1) or 1
            padding = layer.get('padding', 0) or 0
            out_ch = layer['out_ch']
            # Standard conv output spatial size
            H_out = (H + 2 * padding - kernel) // stride + 1
            W_out = (W + 2 * padding - kernel) // stride + 1
            H_out = max(H_out, 1); W_out = max(W_out, 1)
            total_bytes += out_ch * H_out * W_out * bytes_per_element
            # Update H, W for next layer estimate (approximate)
            if stride > 1:
                H, W = H_out, W_out

        for layer in linear_layers:
            total_bytes += layer['out_features'] * bytes_per_element

        for layer in attn_layers:
            # Attention: key/query/value matrices, seq_len proxy from embed_dim
            embed_dim = layer.get('embed_dim', 768)
            seq_len_proxy = 196  # default (14×14 patch grid for ViT 224×224)
            total_bytes += 3 * seq_len_proxy * embed_dim * bytes_per_element

        # Scale by batch size; add 20% overhead for gradient buffers, temp storage
        total_bytes_batched = total_bytes * batch_size * 1.2
        return total_bytes_batched / (1024 ** 2)

    # ── Layer enumeration ─────────────────────────────────────────────────────

    def _enumerate_layers(self, model):
        """Return structured lists of conv, linear, and attention layers."""
        conv_layers   = []
        linear_layers = []
        attn_layers   = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append({
                    'out_ch':  module.out_channels,
                    'in_ch':   module.in_channels,
                    'kernel':  module.kernel_size[0] if isinstance(module.kernel_size, (tuple, list)) else module.kernel_size,
                    'stride':  module.stride[0]  if isinstance(module.stride,  (tuple, list)) else module.stride,
                    'padding': module.padding[0] if isinstance(module.padding, (tuple, list)) else module.padding,
                    'params':  sum(p.numel() for p in module.parameters()),
                })
            elif isinstance(module, nn.Linear):
                linear_layers.append({
                    'in_features':  module.in_features,
                    'out_features': module.out_features,
                    'params':       sum(p.numel() for p in module.parameters()),
                })
            elif isinstance(module, nn.MultiheadAttention):
                attn_layers.append({
                    'embed_dim': module.embed_dim,
                    'num_heads': module.num_heads,
                })

        return conv_layers, linear_layers, attn_layers

    # ── FLOPs estimate ────────────────────────────────────────────────────────

    def _estimate_flops(self, model, input_shape):
        """Fast FLOPs estimate without thop (avoids subprocess overhead)."""
        try:
            from thop import profile
            device = torch.device('cpu')
            model_cpu = model.to(device)
            dummy = torch.randn(1, *input_shape, device=device)
            macs, _ = profile(model_cpu, inputs=(dummy,), verbose=False)
            return int(macs * 2)
        except Exception:
            # Manual fallback: 2 × MACs ≈ 2 × params × spatial_factor
            total_params = sum(p.numel() for p in model.parameters())
            spatial = (input_shape[-2] * input_shape[-1]) if len(input_shape) >= 2 else (224 * 224)
            return int(2 * total_params * (spatial / (224 * 224)) ** 0.5)

    # ── Layer type counting (parallel) ────────────────────────────────────────

    def _count_layer_types(self, model):
        layer_counts = defaultdict(int)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(self._process_module, m) for _, m in model.named_children()]
            for f in concurrent.futures.as_completed(futures):
                for k, v in f.result().items():
                    layer_counts[k] += v
        return dict(layer_counts)

    def _process_module(self, module):
        counts = defaultdict(int)
        counts[module.__class__.__name__] += 1
        for child in module.children():
            counts[child.__class__.__name__] += 1
        return counts

    # ── Architecture patterns ─────────────────────────────────────────────────

    def _extract_architecture_patterns(self, model):
        patterns = {
            'has_skip_connections': False,
            'has_attention':        False,
            'has_normalization':    False,
            'max_depth':            0,
        }
        cls = model.__class__.__name__.lower()
        if 'resnet' in cls or 'densenet' in cls or 'efficientnet' in cls:
            patterns['has_skip_connections'] = True
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'mha' in name.lower():
                patterns['has_attention'] = True
            if isinstance(module, nn.MultiheadAttention):
                patterns['has_attention'] = True
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                patterns['has_normalization'] = True
        patterns['max_depth'] = self._estimate_model_depth(model)
        return patterns

    def _estimate_model_depth(self, model):
        def count_layers(m, d=1):
            mx = d
            for child in m.children():
                mx = max(mx, count_layers(child, d + 1) if list(child.children()) else d + 1)
            return mx
        return count_layers(model)

    # ── Batch analysis ────────────────────────────────────────────────────────

    def analyze_batch(self, models, input_shapes=None):
        if input_shapes is None:
            input_shapes = [(3, 224, 224)] * len(models)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self.extract_features, m, s): i
                       for i, (m, s) in enumerate(zip(models, input_shapes))}
            for f in concurrent.futures.as_completed(futures):
                results.append((futures[f], f.result()))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
