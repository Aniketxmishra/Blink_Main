"""
api/main.py — NeuSight REST API
================================
Exposes NeuSight's prediction and optimization capabilities over HTTP so any
language / service can consume them without importing Python.

Start with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""
from __future__ import annotations

import os
import sys
import time
import logging
from typing import Optional, List

import numpy as np

# Make sure the project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    ModelFeatures, NamedModelRequest, PredictRequest, OptimizeRequest,
    NamedOptimizeRequest, PredictionResponse, OptimizeResponse,
    BatchPoint, HealthResponse,
)

# ─────────────────────────────────────────────────────────────────────────────
# App bootstrap
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("neusight.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="NeuSight GPU Prediction API",
    description=(
        "Predict GPU execution time, memory usage, and optimal batch sizes "
        "for PyTorch models — without running them on a GPU."
    ),
    version="2.0.0",
    contact={"name": "NeuSight", "url": "https://github.com/Aniketxmishra/Blink_Main"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-load heavy dependencies once at startup
# ─────────────────────────────────────────────────────────────────────────────
_predictor = None
_analyzer  = None

def get_predictor():
    global _predictor
    if _predictor is None:
        from gpu_predictor import GPUPredictor
        _predictor = GPUPredictor(
            model_path='models/random_forest_model.joblib',
            memory_model_path='models/memory_model.joblib',
        )
    return _predictor

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from model_analyser import ModelAnalyzer
        _analyzer = ModelAnalyzer()
    return _analyzer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    "resnet18":    lambda: __import__('torchvision').models.resnet18(weights=None),
    "resnet50":    lambda: __import__('torchvision').models.resnet50(weights=None),
    "vgg16":       lambda: __import__('torchvision').models.vgg16(weights=None),
    "mobilenet_v2":lambda: __import__('torchvision').models.mobilenet_v2(weights=None),
    "densenet121": lambda: __import__('torchvision').models.densenet121(weights=None),
    "efficientnet_b0": lambda: __import__('torchvision').models.efficientnet_b0(weights=None),
}

def features_to_dict(f: ModelFeatures) -> dict:
    """Convert Pydantic ModelFeatures -> plain dict for GPUPredictor."""
    d = f.model_dump()
    # Fill optional memory-model fields with heuristics if not provided
    if d.get('weight_memory_mb') is None:
        d['weight_memory_mb'] = d['model_size_mb']
    if d.get('activation_memory_mb') is None:
        # empirical: ~0.5 × weight memory per sample
        d['activation_memory_mb'] = d['model_size_mb'] * 0.5 * d['batch_size']
    if d.get('activation_memory_per_sample') is None:
        d['activation_memory_per_sample'] = d['activation_memory_mb'] / max(d['batch_size'], 1)
    if d.get('flops_per_activation_mb') is None:
        d['flops_per_activation_mb'] = d['flops'] / max(d['activation_memory_mb'], 1e-3)
    if d.get('input_resolution_factor') is None:
        d['input_resolution_factor'] = 224 * 224
    return d


def _load_named_model(model_name: str, input_size: List[int] = None,
                       batch_size: int = 32) -> ModelFeatures:
    """Extract features from a named pretrained model architecture."""
    key = model_name.lower().replace('-', '_')
    if key not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model '{model_name}'. Supported: {list(SUPPORTED_MODELS.keys())}"
        )
    model    = SUPPORTED_MODELS[key]()
    analyzer = get_analyzer()
    inp      = tuple(input_size) if input_size else (3, 224, 224)
    feats    = analyzer.extract_features(model, inp)
    feats['batch_size'] = batch_size
    return ModelFeatures(**{k: feats.get(k) for k in ModelFeatures.model_fields if k in feats})


def _run_prediction(feat_dict: dict) -> PredictionResponse:
    predictor = get_predictor()
    result    = predictor.predict(feat_dict)
    return PredictionResponse(
        batch_size=feat_dict['batch_size'],
        execution_time_ms=result.get('exec_time_ms') or result.get('execution_time_ms', 0.0),
        exec_lower_ms=result.get('exec_lower_ms'),
        exec_upper_ms=result.get('exec_upper_ms'),
        memory_mb=result.get('memory_usage_mb', 0.0),
        memory_lower_mb=result.get('memory_lower_mb'),
        memory_upper_mb=result.get('memory_upper_mb'),
    )


def _run_optimize(feat_dict: dict, min_batch: int, max_batch: int,
                  memory_limit_mb: float) -> OptimizeResponse:
    predictor = get_predictor()
    res = predictor.optimize_batch_size(
        feat_dict, min_batch=min_batch,
        max_batch=max_batch, memory_limit_mb=memory_limit_mb
    )
    if res.get('error'):
        raise HTTPException(status_code=422, detail=res['error'])

    def to_batch_point(r: dict) -> BatchPoint:
        return BatchPoint(
            batch_size=r['batch_size'],
            exec_time_ms=r.get('exec_time_ms', 0.0),
            throughput=r.get('throughput', 0.0),
            memory_usage_mb=r.get('memory_usage_mb', 0.0),
            corrected_memory_mb=r.get('corrected_memory_mb'),
            efficiency=r.get('efficiency'),
            is_pareto=r.get('is_pareto', False),
        )

    return OptimizeResponse(
        optimal_batch_size=res['optimal_batch_size'],
        predicted_execution_time=res.get('predicted_execution_time', 0.0),
        estimated_memory_usage=res.get('estimated_memory_usage', 0.0),
        pareto_front=[to_batch_point(r) for r in res.get('pareto_front', [])],
        all_results=[to_batch_point(r) for r in res.get('batch_results', [])],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Request timing middleware
# ─────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {"message": "NeuSight GPU Prediction API v2 — see /docs for usage"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    Health check — confirms models are loaded and optionally reports GPU.
    """
    predictor = get_predictor()
    gpu_name  = None
    has_gpu   = False
    try:
        import pynvml
        pynvml.nvmlInit()
        h        = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(h)
        has_gpu  = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        exec_model=True,
        memory_model=predictor.has_memory_model,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
    )


@app.get("/models", tags=["System"])
def list_models():
    """List supported pretrained model names for /predict/named and /optimize/named."""
    return {"supported_models": list(SUPPORTED_MODELS.keys())}


# ── Prediction endpoints ──────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(body: PredictRequest):
    """
    Predict execution time and memory usage from raw model features.

    Use this when you've already run `ModelAnalyzer.extract_features()` locally
    and just want the server to run inference on its trained models.
    """
    feat_dict = features_to_dict(body.features)
    return _run_prediction(feat_dict)


@app.post("/predict/named", response_model=PredictionResponse, tags=["Prediction"])
def predict_named(body: NamedModelRequest):
    """
    Predict execution time and memory for a named pretrained architecture.
    The server loads the model, extracts features, and returns predictions.

    Supported: resnet18, resnet50, vgg16, mobilenet_v2, densenet121, efficientnet_b0
    """
    feat_obj  = _load_named_model(body.model_name, body.input_size, body.batch_size)
    feat_dict = features_to_dict(feat_obj)
    return _run_prediction(feat_dict)


# ── Optimization endpoints ────────────────────────────────────────────────────

@app.post("/optimize", response_model=OptimizeResponse, tags=["Optimization"])
def optimize(body: OptimizeRequest):
    """
    Find the Pareto-optimal batch size for a model described by raw features.

    Returns:
    - `optimal_batch_size`: best efficiency (throughput / memory) recommendation
    - `pareto_front`: non-dominated batch sizes — use these to make your own
      trade-off between throughput and memory footprint
    - `all_results`: full sweep of all tested batch sizes
    """
    feat_dict = features_to_dict(body.features)
    return _run_optimize(feat_dict, body.min_batch, body.max_batch, body.memory_limit_mb)


@app.post("/optimize/named", response_model=OptimizeResponse, tags=["Optimization"])
def optimize_named(body: NamedOptimizeRequest):
    """
    Find the Pareto-optimal batch size for a named pretrained architecture.

    Example use-case: before starting a training job, call this endpoint to
    know the safest maximum batch size that fits within your GPU's VRAM budget.
    """
    feat_obj  = _load_named_model(body.model_name, body.input_size, 1)
    feat_dict = features_to_dict(feat_obj)
    return _run_optimize(feat_dict, body.min_batch, body.max_batch, body.memory_limit_mb)


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry-point  (python api/main.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
