"""
model_analyser.py  —  Activation-Aware Model Feature Extractor

Updated (Priority 2) to support HuggingFace Causal LMs (GPT-2, LLaMA, Mistral)
in addition to PyTorch Vision CNNs and ViTs.

Key additions:
  - Detects `transformers.PreTrainedModel` instances automatically.
  - Extracts vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
    intermediate_size from the model config.
  - Computes physically-grounded KV Cache memory for a given seq_len:
      kv_cache_bytes = 2 × batch × seq_len × num_heads × head_dim × bytes_per_param × num_layers
  - Separates prefill and decode memory estimates for autoregressive models.
  - Quantization-aware: scales bytes_per_param for fp16, int8, and int4.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict
import glob
import os
import json
import concurrent.futures
from sklearn.model_selection import train_test_split, KFold, GroupShuffleSplit
import xgboost as xgb


class ModelAnalyzer:
    """Scalable model architecture analyzer with activation-aware feature extraction."""

    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    # ── Public API ───────────────────────────────────────────────────────────

    def extract_features(self, model, input_shape=(3, 224, 224), batch_size=1,
                         seq_len=128, quantization_bits=32):
        """Extract all features needed by the prediction and memory models.

        Args:
            model: PyTorch nn.Module OR a HuggingFace PreTrainedModel.
            input_shape: (C, H, W) — single-sample input dimensions (ignored for LLMs).
            batch_size: used to scale activation and KV cache memory estimates.
            seq_len: token sequence length (relevant for LLMs, ignored for CNNs).
            quantization_bits: precision bits — 32 (fp32), 16 (fp16), 8 (int8), or 4 (int4).
                               Scales model weight memory and KV cache accordingly.
        """
        # ── Detect HuggingFace LLM ────────────────────────────────────────────
        is_llm = self._is_huggingface_llm(model)
        if is_llm:
            return self._extract_llm_features(model, batch_size, seq_len, quantization_bits)

        # ── Basic params (Vision / CNN path) ─────────────────────────────────
        total_params      = sum(p.numel() for p in model.parameters())
        trainable_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)

        param_size   = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size  = sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)

        # Scale weight memory by quantization (fp32 baseline)
        quant_scale   = self._quantization_memory_scale(quantization_bits)
        weight_memory_mb = model_size_mb * quant_scale

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

            # Activation-aware
            'activation_memory_mb':         activation_memory_mb,
            'weight_memory_mb':             weight_memory_mb,
            'activation_memory_per_sample': activation_memory_per_sample,
            'flops_per_activation_mb':      flops_per_activation_mb,
            'input_resolution_factor':      input_resolution_factor,

            # Compute
            'flops':                   flops,
            'compute_memory_ratio':    compute_memory_ratio,
            'memory_read_write_ratio': 0.5,

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

            # LLM-specific (zero for Vision models)
            'vocab_size':          0,
            'hidden_size':         0,
            'num_hidden_layers':   0,
            'num_attention_heads': 0,
            'intermediate_size':   0,
            'seq_len':             0,
            'kv_cache_size_mb':    0.0,
            'prefill_memory_mb':   0.0,
            'decode_memory_mb':    0.0,
            'quantization_bits':   quantization_bits,
            'is_llm':              False,

            # Patterns
            'layer_counts':           layer_counts,
            'architecture_patterns':  architecture_patterns,
        }

    # ── LLM detection & feature extraction ───────────────────────────────────

    def _is_huggingface_llm(self, model):
        """Check if the model is a HuggingFace PreTrainedModel."""
        try:
            from transformers import PreTrainedModel
            return isinstance(model, PreTrainedModel)
        except ImportError:
            return False

    @staticmethod
    def _quantization_memory_scale(quantization_bits):
        """Return the memory scaling factor relative to fp32 baseline."""
        scales = {32: 1.0, 16: 0.5, 8: 0.25, 4: 0.125}
        return scales.get(quantization_bits, 1.0)

    def _extract_llm_features(self, model, batch_size, seq_len, quantization_bits):
        """Extract features specific to HuggingFace Causal/Seq2Seq LLMs.

        Key computed metrics:
          - kv_cache_size_mb:  KV Cache for the FULL sequence (prefill phase).
          - prefill_memory_mb: Peak memory when processing the whole prompt at once.
          - decode_memory_mb:  Peak memory per decode step (1 new token, KV cached).
        """
        cfg = getattr(model, 'config', None)

        # Pull standard HF config fields (with safe fallbacks)
        vocab_size          = getattr(cfg, 'vocab_size',          50257)
        hidden_size         = getattr(cfg, 'hidden_size',         getattr(cfg, 'd_model', 768))
        num_hidden_layers   = getattr(cfg, 'num_hidden_layers',   getattr(cfg, 'n_layer', 12))
        num_attention_heads = getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_head',  12))
        intermediate_size   = getattr(cfg, 'intermediate_size',   hidden_size * 4)
        head_dim            = hidden_size // max(num_attention_heads, 1)

        # Basic params
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_size       = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size      = sum(b.nelement() * b.element_size() for b in model.buffers())
        model_size_mb    = (param_size + buffer_size) / (1024 ** 2)

        quant_scale      = self._quantization_memory_scale(quantization_bits)
        weight_memory_mb = model_size_mb * quant_scale

        # ── KV Cache Memory (physically-grounded) ─────────────────────────────
        # Each layer stores Key and Value tensors:
        #   shape = [batch, num_heads, seq_len, head_dim]
        # Memory = 2 (K+V) × batch × seq_len × num_heads × head_dim × bytes_per_element × num_layers
        bytes_per_element = 4 * quant_scale   # fp32 baseline, scaled by quantization
        kv_cache_bytes = (
            2 * batch_size * seq_len * num_attention_heads * head_dim
            * bytes_per_element * num_hidden_layers
        )
        kv_cache_size_mb = kv_cache_bytes / (1024 ** 2)

        # ── Prefill memory: weight memory + activation for full prompt ─────────
        # Activation during prefill ≈ seq_len × hidden_size × 4 bytes × num_layers (simplified)
        prefill_activation_bytes = seq_len * hidden_size * 4 * num_hidden_layers * batch_size
        prefill_memory_mb = weight_memory_mb + (prefill_activation_bytes / (1024 ** 2)) + kv_cache_size_mb

        # ── Decode memory: weight memory + full KV cache (growing each step) ───
        # During decode, the model sees only 1 new token but must hold the full KV cache
        decode_activation_bytes = 1 * hidden_size * 4 * num_hidden_layers * batch_size  # 1 new token
        decode_memory_mb = weight_memory_mb + kv_cache_size_mb + (decode_activation_bytes / (1024 ** 2))

        # ── FLOPs estimate for a single forward pass ──────────────────────────
        # Prefill FLOPs: 2 * seq_len * (4 * hidden^2 * num_layers) — standard transformer estimate
        flops = int(2 * seq_len * 4 * hidden_size ** 2 * num_hidden_layers)

        compute_memory_ratio = flops / max(total_params * bytes_per_element, 1)

        return {
            # Basic
            'total_parameters':        total_params,
            'trainable_parameters':    trainable_params,
            'model_size_mb':           model_size_mb,

            # Memory
            'weight_memory_mb':             weight_memory_mb,
            'activation_memory_mb':         prefill_memory_mb,  # use prefill as general activation proxy
            'activation_memory_per_sample': prefill_memory_mb / max(batch_size, 1),
            'flops_per_activation_mb':      flops / max(prefill_memory_mb, 1e-3),
            'input_resolution_factor':      seq_len,  # seq_len is the LLM equivalent of spatial resolution

            # Compute
            'flops':                   flops,
            'compute_memory_ratio':    compute_memory_ratio,
            'memory_read_write_ratio': 0.5,

            # Layer structure (mostly zero for LLMs — no conv)
            'num_conv_layers':     0,
            'num_fc_layers':       num_hidden_layers * 4,  # approximate (Q, K, V, Out proj per layer)
            'num_bn_layers':       num_hidden_layers,       # LayerNorms
            'num_attn_layers':     num_hidden_layers,
            'avg_conv_kernel_size':0.0,
            'max_conv_channels':   0,
            'total_conv_params':   0,
            'max_fc_size':         intermediate_size,
            'total_fc_params':     total_params,
            'model_depth':         num_hidden_layers,

            # LLM-specific (the KEY new features)
            'vocab_size':          vocab_size,
            'hidden_size':         hidden_size,
            'num_hidden_layers':   num_hidden_layers,
            'num_attention_heads': num_attention_heads,
            'intermediate_size':   intermediate_size,
            'seq_len':             seq_len,
            'kv_cache_size_mb':    kv_cache_size_mb,
            'prefill_memory_mb':   prefill_memory_mb,
            'decode_memory_mb':    decode_memory_mb,
            'quantization_bits':   quantization_bits,
            'is_llm':              True,

            # Patterns
            'layer_counts':           {'Linear': num_hidden_layers * 4, 'LayerNorm': num_hidden_layers},
            'architecture_patterns':  {
                'has_skip_connections': True,
                'has_attention':        True,
                'has_normalization':    True,
                'max_depth':            num_hidden_layers,
            },
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

    # ── Interval / Uncertainty Bounds Evaluation ──────────────────────────────
    @staticmethod
    def interval_pinball_loss(y_true, q_lower, q_upper, alpha=0.2):
        """
        Compute the interval loss (pinball loss equivalent for bounds) for OOD validation.
        Formula: (upper - lower) + (2/alpha) * max(0, lower - y) + (2/alpha) * max(0, y - upper)
        Here, alpha = 0.2 represents an 80% CI (tau=0.1, tau=0.9).
        """
        q_lower = np.array(q_lower)
        q_upper = np.array(q_upper)
        y_true = np.array(y_true)
        
        width = q_upper - q_lower
        penalty_lower = (2.0 / alpha) * np.maximum(0, q_lower - y_true)
        penalty_upper = (2.0 / alpha) * np.maximum(0, y_true - q_upper)
        
        return np.mean(width + penalty_lower + penalty_upper)

    @staticmethod
    def ood_extrapolation_split(df, mode='batch_size', train_thresh=8):
        """
        Generate explicit OOD splits to prove the predictor explicitly works on unseen data.
        mode='batch_size': Train on BS <= thresh, test on BS > thresh
        mode='family': Group by model family (e.g. ResNext, VGG) and do LOFO split.
        """
        if mode == 'batch_size':
            train = df[df['batch_size'] <= train_thresh]
            test = df[df['batch_size'] > train_thresh]
            return train, test
            
        elif mode == 'family':
            # Identify pseudo families
            df['family'] = df['model_name'].str.split('_').str[0].str.extract(r'([a-zA-Z]+)')[0]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(df, groups=df['family']))
            return df.iloc[train_idx], df.iloc[test_idx]
        
        return train_test_split(df, test_size=0.2, random_state=42)
