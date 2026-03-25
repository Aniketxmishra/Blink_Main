# Blink 🔭

[![PyPI version](https://badge.fury.io/py/blink-gpu.svg)](https://badge.fury.io/py/blink-gpu)
[![CI](https://github.com/Aniketxmishra/Blink_Main/actions/workflows/ci.yml/badge.svg)](https://github.com/Aniketxmishra/Blink_Main/actions/workflows/ci.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **GPU Performance Predictor for Deep Learning Models**

**Blink** predicts the **execution time** and **peak memory usage** of PyTorch neural networks on GPU hardware *before* you actually run or deploy them. 

It combines classical ML (XGBoost, Random Forest) with a Graph Neural Network (GNN) that encodes the computational graph of any model architecture, acting as a "virtual profiler."

---

## ⚡ Quick Start

### Installation

Blink is published on PyPI. You can install the core API, or install with optional dependency groups:

```bash
# Core prediction API only
pip install blink-gpu

# Include Streamlit Dashboard, SHAP explainability, and Plotly
pip install "blink-gpu[full]"

# Include FastAPI REST Server
pip install "blink-gpu[api]"

# Install everything
pip install "blink-gpu[all]"
```

*Note: You must install PyTorch (`torch`, `torchvision`) separately according to your CUDA hardware.*

### Python Usage

```python
import torchvision.models as tv
from blink import BlinkPredictor, BlinkAnalyzer

# 1. Analyze any PyTorch model architecture
model = tv.resnet18(weights=None)
print(BlinkAnalyzer().summary(model))
# ➔ Parameters: 11,689,512 | FLOPs: 1,814 M | Conv layers: 20 | Size: 44.59 MB

# 2. Predict execution time and memory for a batch size
predictor = BlinkPredictor()
result = predictor.predict(model, batch_size=32)

print(f"Exec time: {result['exec_time_ms']:.1f} ms")
print(f"Memory   : {result['memory_mb']:.1f} MB")
# ➔ Exec time: 18.3 ms | Memory: 184.3 MB

# 3. Sweep multiple batch sizes
sweep = predictor.predict_batch("resnet50", batch_sizes=[1, 16, 32, 64])
```

---

## 💻 Command Line Interface (CLI)

Blink comes with a built-in CLI for quick profiling without writing scripts:

```bash
# Predict via CLI
$ blink predict resnet50 --batch-size 32
🔮 Blink prediction for 'resnet50'
 Batch   Exec (ms)   Memory (MB)  CI-Exec (80%)
------------------------------------------------------------
    32       28.45         294.5  [22.1 - 36.6]

# Launch the Streamlit Dashboard
$ blink dashboard --port 8501

# Launch the FastAPI REST Server
$ blink server --host 0.0.0.0 --port 8000
```

---

## 📊 Streamlit Dashboard & Explainability

Blink includes a rich, interactive web dashboard. Run `blink dashboard` to access:

![Blink Dashboard SHAP Explainability Demo](results/figures/dashboard_shap_demo.png)

- **Live Predictions:** Instantly predict performance for custom PyTorch code or TorchVision models.
- **🔍 SHAP Explainability ("Why this prediction?"):** Interactive waterfall charts explaining exactly *which architectural features* (e.g., FLOPs, Conv layers, Model Depth) drove the predicted execution time and memory footprint up or down.

![Blink Batch Optimizer Demo](results/figures/dashboard_batch_optimizer.png)

- **Batch Size Optimizer:** Find the maximum batch size that fits within your specific GPU memory budget (e.g., 8GB, 16GB, 24GB).
- **Compare Architectures:** Side-by-side performance comparison of different models.

---

## 🌐 REST API & Docker Deployment

Blink can be deployed as a microservice to provide GPU cost estimates to other applications.

### Docker Compose (Recommended)
You can spin up both the Streamlit Dashboard and the FastAPI backend instantly using Docker.

```bash
git clone https://github.com/Aniketxmishra/Blink_Main.git
cd Blink_Main
docker compose up -d
```
- **Dashboard:** `http://localhost:8501`
- **REST API:** `http://localhost:8000/docs` (Swagger UI)

### REST API Example
```bash
curl -X POST "http://localhost:8000/api/v2/predict" \
     -H "Content-Type: application/json" \
     -d '{"model_name": "resnet50", "batch_size": 32}'

# Response:
# {
#   "model_name": "resnet50",
#   "batch_size": 32,
#   "predictions": {
#     "exec_time_ms": 28.45,
#     "exec_time_bounds": [22.1, 36.6],
#     "memory_usage_mb": 294.5,
#     ...
#   }
# }
```

---

## 🧠 How it Works (Architecture)

```text
PyTorch Model
      │
      ▼
┌─────────────────────┐
│  Feature Extractor  │  ← layer counts, FLOPs, params, depth, width, skip connections
│  + GNN Extractor    │  ← graph-based architecture encoding (ArchitectureGNN)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Prediction Models  │
│  ─────────────────  │
│  · XGBoost (tuned)  │  ← main predictor (best MAPE) + SHAP Explainer
│  · Random Forest    │  ← latency confidence intervals (Quantile Regression)
│  · GNN Predictor    │  ← graph-native, generalizes across architectures
└─────────┬───────────┘
          │
          ▼
   Predicted: exec_time_ms, memory_mb
```

**Model Performance on Held-out Data:**
- **Execution Time (XGBoost):** ~8% MAPE
- **Memory Usage (XGBoost):** ~6% MAPE

---

## 🔬 Development & Paper Reproducibility

Blink was developed alongside a research study evaluating the efficacy of static and graph-based features for GPU performance prediction.

**To reproduce the paper's figures and ablation study:**
```bash
git clone https://github.com/Aniketxmishra/Blink_Main.git
cd Blink_Main
pip install -e ".[full]"

python scripts/ablation_study.py
python scripts/generate_paper_figures.py
```
*Outputs will be saved to the `results/` directory.*

---

## 📄 License
MIT License — see [LICENSE](LICENSE) for details. Made by [Aniket Mishra](https://github.com/Aniketxmishra).
