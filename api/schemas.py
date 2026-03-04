"""
api/schemas.py — Pydantic request / response models for NeuSight REST API
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class ModelFeatures(BaseModel):
    """
    Raw architecture features extracted from a PyTorch model.
    All fields match the columns produced by ModelAnalyzer.extract_features().
    Pass these when you have already run the analyser locally, or when you
    cannot ship the weights to the server.
    """
    batch_size:             int   = Field(32,   description="Inference batch size")
    flops:                  float = Field(...,  description="Total FLOPs (floating-point ops)")
    compute_memory_ratio:   float = Field(...,  description="FLOPs / model_size_mb")
    num_conv_layers:        int   = Field(0)
    num_fc_layers:          int   = Field(0)
    num_bn_layers:          int   = Field(0)
    avg_conv_kernel_size:   float = Field(3.0)
    max_conv_channels:      int   = Field(0)
    total_conv_params:      int   = Field(0)
    total_fc_params:        int   = Field(0)
    model_depth:            int   = Field(0)
    model_size_mb:          float = Field(...,  description="Model weight size in MB")
    # memory-model extras (optional — filled with heuristics if absent)
    activation_memory_mb:         Optional[float] = None
    weight_memory_mb:             Optional[float] = None
    activation_memory_per_sample: Optional[float] = None
    flops_per_activation_mb:      Optional[float] = None
    input_resolution_factor:      Optional[float] = 50176.0  # 224*224


class NamedModelRequest(BaseModel):
    """Use a well-known pretrained architecture by name instead of raw features."""
    model_name: str = Field(..., description="One of: resnet18, resnet50, vgg16, mobilenet_v2, densenet121")
    batch_size:  int = Field(32)
    input_size:  List[int] = Field([3, 224, 224], description="CHW input tensor shape")


class PredictRequest(BaseModel):
    """Predict execution time + memory for a given set of model features."""
    features: ModelFeatures


class OptimizeRequest(BaseModel):
    """Find the Pareto-optimal batch size for a given model."""
    features:        ModelFeatures
    min_batch:       int   = Field(1,    ge=1)
    max_batch:       int   = Field(128,  le=1024)
    memory_limit_mb: float = Field(8000, description="Available VRAM in MB")


class NamedOptimizeRequest(BaseModel):
    """Like OptimizeRequest but uses a named pretrained model."""
    model_name:      str   = Field(...)
    input_size:      List[int] = Field([3, 224, 224])
    min_batch:       int   = Field(1,    ge=1)
    max_batch:       int   = Field(128,  le=1024)
    memory_limit_mb: float = Field(8000)


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    batch_size:           int
    execution_time_ms:    float
    exec_lower_ms:        Optional[float]
    exec_upper_ms:        Optional[float]
    memory_mb:            float
    memory_lower_mb:      Optional[float]
    memory_upper_mb:      Optional[float]
    model_version:        str = "v2"


class BatchPoint(BaseModel):
    batch_size:           int
    exec_time_ms:         float
    throughput:           float
    memory_usage_mb:      float
    corrected_memory_mb:  Optional[float]
    efficiency:           Optional[float]
    is_pareto:            bool = False


class OptimizeResponse(BaseModel):
    optimal_batch_size:        int
    predicted_execution_time:  float
    estimated_memory_usage:    float
    pareto_front:              List[BatchPoint]
    all_results:               List[BatchPoint]


class HealthResponse(BaseModel):
    status:        str
    exec_model:    bool
    memory_model:  bool
    has_gpu:       bool
    gpu_name:      Optional[str]
