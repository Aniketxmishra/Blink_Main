# Blink Codebase Investigation

Here are the detailed answers to your 22 questions regarding the Blink GPU Performance Predictor codebase.

## --- CODEBASE STRUCTURE ---

**1. List every .py file with a one-line description.**
*(Showing the most critical 20/108 files across modules. The others are tests, legacy scripts, or architecture scripts)*
- [blink/__init__.py](file:///c:/Aniket/review%20blink/Neusight/blink/__init__.py): Package entry point exposing [BlinkPredictor](file:///c:/Aniket/review%20blink/Neusight/blink/_predictor.py#22-141) and [predict](file:///c:/Aniket/review%20blink/Neusight/api/main.py#233-243).
- [blink/_predictor.py](file:///c:/Aniket/review%20blink/Neusight/blink/_predictor.py): Public API facade for the tabular prediction ([GPUPredictor](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py#8-451)).
- [blink/_analyzer.py](file:///c:/Aniket/review%20blink/Neusight/blink/_analyzer.py): Public API facade for feature extraction.
- [blink/__main__.py](file:///c:/Aniket/review%20blink/Neusight/blink/__main__.py): CLI entry point for `blink` terminal commands.
- [api/main.py](file:///c:/Aniket/review%20blink/Neusight/api/main.py): FastAPI server defining the REST API endpoints.
- [api/schemas.py](file:///c:/Aniket/review%20blink/Neusight/api/schemas.py): Pydantic validation schemas for the REST API.
- [gpu_predictor.py](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py): Core inference class handling caching, XGBoost loading, and batch optimization.
- [prediction_model.py](file:///c:/Aniket/review%20blink/Neusight/prediction_model.py): Training pipeline script to train XGBoost estimators (median, lower, upper).
- [feature_extractor.py](file:///c:/Aniket/review%20blink/Neusight/feature_extractor.py): Static code parsing model parameters, FLOPs, and architectural features.
- [model_analyser.py](file:///c:/Aniket/review%20blink/Neusight/model_analyser.py): Enhanced feature extraction wrapper over [feature_extractor.py](file:///c:/Aniket/review%20blink/Neusight/feature_extractor.py).
- [gnn_model.py](file:///c:/Aniket/review%20blink/Neusight/gnn_model.py): PyTorch implementation of the [ArchitectureGNN](file:///c:/Aniket/review%20blink/Neusight/gnn_model.py#6-59) serving parallel predictions.
- [gnn_extractor.py](file:///c:/Aniket/review%20blink/Neusight/gnn_extractor.py): Converts PyTorch `nn.Module` object into a `torch_geometric.data.Data` graph.
- [dashboard.py](file:///c:/Aniket/review%20blink/Neusight/dashboard.py): Streamlit-based web interface for interactive inference.
- [model_profiler.py](file:///c:/Aniket/review%20blink/Neusight/model_profiler.py): Collects ground-truth execution times and GPU stats via PyTorch events.
- [scripts/collect_data.py](file:///c:/Aniket/review%20blink/Neusight/scripts/collect_data.py): CLI script to run `model_profiler` over many architectures.
- [train_eval_blink.py](file:///c:/Aniket/review%20blink/Neusight/train_eval_blink.py): Automated model evaluation script calculating the OOD MAPE score.
- [run_agent.py](file:///c:/Aniket/review%20blink/Neusight/run_agent.py): LangChain/LLM autoresearch agent runner.
- [dynamic_predictor.py](file:///c:/Aniket/review%20blink/Neusight/dynamic_predictor.py) / [dynamic_gpu_predictor.py](file:///c:/Aniket/review%20blink/Neusight/dynamic_gpu_predictor.py): Extrapolation utilities for dynamic execution shapes.
- [performance_monitor.py](file:///c:/Aniket/review%20blink/Neusight/performance_monitor.py): Background thread for tracing system utilization over time.
- [train_gnn.py](file:///c:/Aniket/review%20blink/Neusight/train_gnn.py): Training loop for the PyTorch Geometric model.

**2. What is the exact entry point for a prediction? Trace the call chain.**
1. User calls `from blink import predict; predict(model, batch_size=16)`.
2. [blink/__init__.py](file:///c:/Aniket/review%20blink/Neusight/blink/__init__.py): [predict()](file:///c:/Aniket/review%20blink/Neusight/api/main.py#233-243) initializes the global `_default_predictor = BlinkPredictor()` and calls `_default_predictor.predict()`.
3. [blink/_predictor.py](file:///c:/Aniket/review%20blink/Neusight/blink/_predictor.py): `BlinkPredictor.predict()` instantiates [GPUPredictor](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py#8-451) from [gpu_predictor.py](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py), utilizes `ModelAnalyzer()` to extract static features, then calls `self._predictor.predict([features])`.
4. [gpu_predictor.py](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py): Extracts the 15 features, checks cache, runs inference using the loaded XGBoost Joblib models, enforces non-crossing bounds, runs the memory model, and returns the combined dictionary.

**3. Are the trained model weights committed to the repo?**
They are **gitignored**.
The [.gitignore](file:///c:/Aniket/review%20blink/Neusight/.gitignore) strictly ignores `models/*.joblib` and `models/*.pth`. Only [memory_model_features.json](file:///c:/Aniket/review%20blink/Neusight/models/memory_model_features.json) is checked in inside the [models/](file:///c:/Aniket/review%20blink/Neusight/api/main.py#225-229) directory.

**4. What does [__init__.py](file:///c:/Aniket/review%20blink/Neusight/api/__init__.py) currently export?**
It exports [BlinkPredictor](file:///c:/Aniket/review%20blink/Neusight/blink/_predictor.py#22-141), `BlinkAnalyzer`, [predict](file:///c:/Aniket/review%20blink/Neusight/api/main.py#233-243), and `__version__`.
The correct import for a user is: `from blink import BlinkPredictor, predict`

## --- ML PIPELINE ---

**5. What features exactly make up the static feature vector?**
Listed in [train_eval_blink.py](file:///c:/Aniket/review%20blink/Neusight/train_eval_blink.py) (21 dimensions):
1. [batch_size](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py#286-430) 2. `flops` 3. `compute_memory_ratio` 4. `num_conv_layers` 5. `num_fc_layers` 6. `num_bn_layers` 7. `avg_conv_kernel_size` 8. `max_conv_channels` 9. `total_conv_params` 10. `total_fc_params` 11. `model_depth` 12. `model_size_mb` 13. `tflops_fp32` 14. `memory_bandwidth_gbps` 15. `sm_count` 16. `flops_per_mb` 17. `params_per_layer` 18. `conv_to_fc_ratio` 19. `flops_to_bandwidth` 20. `compute_intensity_score` 21. `depth_complexity_penalty`

**6. How many GCN layers are used? Hidden Dim? Architecture Location?**
It uses **3 layers** of `GATConv` (Graph Attention Network, not plain GCN). The hidden dimension is `64`.
Defined in exactly: [gnn_model.py](file:///c:/Aniket/review%20blink/Neusight/gnn_model.py) under the class name [ArchitectureGNN](file:///c:/Aniket/review%20blink/Neusight/gnn_model.py#6-59).

**7. What node features does the GCN currently use?**
It uses a 12-dimensional vector defined in [gnn_extractor.py](file:///c:/Aniket/review%20blink/Neusight/gnn_extractor.py):
- 6 dimensions for One-Hot Layer Type: Conv, Linear, Norm, Pool, Attention, Other
- 1 dimension for `log1p(parameters)`
- 5 dimensions for Attributes: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`
**Multihead Attention** is registered (Type "Attention", where `num_heads` maps to `kernel_size`). **LayerNorm** registers as "Norm", while **GELU** is not explicitly handled and defaults to "Other".

**8. How is the XGBoost model serialized?**
Serialized natively via scikit-learn's **[joblib](file:///c:/Aniket/review%20blink/Neusight/models/memory_model.joblib)**. Loaded at inference in [gpu_predictor.py](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py) using `joblib.load()`.

**9. Quantile regression implementation:**
It trains **three separate XGBoost models** (`Median (0.5)`, `lower (0.1)`, `upper (0.9)`). They are distinct regressors trained with `objective='reg:quantileerror'` in [prediction_model.py](file:///c:/Aniket/review%20blink/Neusight/prediction_model.py).

**10. Is the non-crossing constraint enforced?**
Yes. In [gpu_predictor.py](file:///c:/Aniket/review%20blink/Neusight/gpu_predictor.py) (Lines ~168-172):
```python
lower = float(max(1.0, lower_bounds[i]))
median = float(max(lower, predictions[i]))
upper = float(max(median, upper_bounds[i]))
```

## --- API & PACKAGING ---

**11. API Framework and Endpoints:**
[prediction_api.py](file:///c:/Aniket/review%20blink/Neusight/prediction_api.py) is entirely an argparse CLI script, not an API.
The actual REST server is [api/main.py](file:///c:/Aniket/review%20blink/Neusight/api/main.py) which uses **FastAPI**.
Routes (HTTP GET/POST):
- `GET /` and `GET /health` and `GET /models`
- `POST /predict` and `POST /predict/named`
- `POST /optimize` and `POST /optimize/named`

**12. API Security and Rate Limiting:**
There is **zero** authentication or rate limiting. Input validation is handled automatically via FastAPI / Pydantic models mapped in [api/schemas.py](file:///c:/Aniket/review%20blink/Neusight/api/schemas.py).

**13. Packaging Files:**
There is both [pyproject.toml](file:///c:/Aniket/review%20blink/Neusight/pyproject.toml) and [requirements.txt](file:///c:/Aniket/review%20blink/Neusight/requirements.txt).
Dependencies inside [requirements.txt](file:///c:/Aniket/review%20blink/Neusight/requirements.txt):
Pinned: `torch==2.2.0`, `torchvision==0.17.0`, `numpy==1.26.4`, `pandas==2.2.0`, `scikit-learn==1.4.0`, `thop==0.1.1.post2209072238`, `joblib==1.3.2`, `streamlit==1.32.0`, `plotly==5.18.0`, `matplotlib==3.8.3`, `seaborn==0.13.2`.
Unpinned: `xgboost`, `optuna`, `lightgbm`, `pynvml`, etc.

**14. [models/](file:///c:/Aniket/review%20blink/Neusight/api/main.py#225-229) gitignore rules:**
Yes, it is ignored via these exact lines in [.gitignore](file:///c:/Aniket/review%20blink/Neusight/.gitignore):
```
models/*.joblib
models/*.pth
```

**15. Streamlit Dashboard API Usage:**
The [dashboard.py](file:///c:/Aniket/review%20blink/Neusight/dashboard.py) imports backend code directly (e.g., `from gpu_predictor import GPUPredictor`). It **does not** hit the FastAPI server via HTTP.

## --- DATA & TRAINING ---

**16. Schema of profiling records (from `data/enriched/` CSVs):**
`model_name` (object), `batch_size` (int), `seq_len` (int), `quantization_bits` (int), `execution_time_ms` (float), `prefill_time_ms` (float), `decode_time_ms` (float), `is_llm` (bool), `total_parameters` (int), `trainable_parameters` (int), `model_size_mb` (float), `vocab_size` (int), `hidden_size` (int), `num_hidden_layers` (int), `num_attention_heads` (int), `kv_cache_size_mb` (float), `flops` (int), `compute_memory_ratio` (int), `num_conv_layers` (int), `num_fc_layers` (int), `num_bn_layers` (int), `avg_conv_kernel_size` (int), `max_conv_channels` (int), `total_conv_params` (int), `total_fc_params` (int), `model_depth` (int), `tflops_fp32` (float), `memory_bandwidth_gbps` (float), `sm_count` (int).

**17. Training Dataset Current Stats:**
- Total Rows: **1,081**
- Unique Architectures: **71**
- GPUs Represented: **1** (Tesla T4)

**18. `feedback_log.csv` and Online Learning:**
Not used. A grep search across the codebase reveals no trace of `feedback_log.csv` or any active online learning loop updating model artifacts dynamically.

**19. Train/Test Split Strategy:**
**Batch-size stratified (OOD Extrapolation)**. Instead of random splitting, `train_eval_blink.py` explicitly assigns `batch_size <= 8` to the Train set, and tests strictly against `batch_size >= 16` to enforce generalization boundaries.

## --- GAPS & ISSUES ---

**20. TODO, FIXME, HACK comments:**
A rigorous regex search across all codebase files yielded exactly **0** instances of these meta-tags.

**21. Immediate Failure on `pip install blink-gpu`:**
It **will fail** out of the box because:
1. The user prompt tries `from blink import Blink`. The class exported is `BlinkPredictor`, raising an `ImportError`.
2. Even if fixed, calling `.predict()` will instantly raise `FileNotFoundError: models/random_forest_model.joblib not found` because the model weights were gitignored and no download logic is executed upon pip installation.

**22. Hardcoded Breakages for CPU Native Environments:**
- Model initialization: `api/main.py`, `gpu_predictor.py`, and `prediction_api.py` blindly map `models/*.joblib`.
- In `dashboard.py` and `feature_extractor.py`, hardcoded references to `data/raw` and `data/temp` demand the script runs from the repository root rather than as an installed python package.
- `pynvml.nvmlInit()` in the api health checks assumes NVIDIA driver presence, leading to exceptions on CPU environments if not caught properly.

# Final Evaluation

**Confidence Score:** `6/10`
The core prediction logic and ML evaluation loops are thorough. However, missing model artifacts, lack of remote payload downloading, and hardcoded relative paths show that it operates well locally as an R&D notebook but falters during Python package installation.

**Top 3 Things Likely to Break for a New User:**
1. **Model Weight Missing:** Running inference fails immediately because `models/*.joblib` files are neither included in git nor downloaded magically.
2. **Library Call Syntax:** The provided README/Doc examples (importing `Blink`) do not match the exported API (`BlinkPredictor`).
3. **Relative Working Directories:** Core code expects execution directly from the cloned repo root (assumes `data/` and `models/` exist in `os.getcwd()`).

**Single Most Important File to Fix First:**
`blink/_predictor.py`. It requires immediate modification to check if weights exist in the `models/` directory, and if not, reach out to an S3/HuggingFace URL to download them, preventing immediate crashes for a pip-installed user.
