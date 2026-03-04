# NeuSight 🔭
> **GPU Performance Predictor for Deep Learning Models**

NeuSight predicts **execution time** and **memory usage** of PyTorch neural networks on GPU without actually running them. It combines classical ML (XGBoost, Random Forest) with a Graph Neural Network (GNN) that encodes the computational graph of any model architecture.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Model Performance](#model-performance)
- [Dashboard](#dashboard)
- [Paper Reproducibility](#paper-reproducibility)

---

## Overview

Given a PyTorch model and a batch size, NeuSight answers:
- *How long will a forward pass take on this GPU?*
- *How much GPU memory will it consume?*

This is useful for:
- **Batch size optimization** before deployment
- **Hardware cost estimation** for training runs
- **NAS (Neural Architecture Search)** — filtering architectures by predicted cost

---

## Architecture

```
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
│  · XGBoost (tuned)  │  ← main predictor (best MAPE)
│  · Random Forest    │  ← ensemble comparison
│  · GNN Predictor    │  ← graph-native, generalizes across architectures
│  · Linear / Ridge   │  ← baselines
└─────────┬───────────┘
          │
          ▼
   Predicted: execution_time_ms, memory_mb
   + Uncertainty bounds (lower / upper)
```

---

## Project Structure

```
neusight/
├── dashboard.py             # 🖥️  Main Streamlit web app  (run this)
├── prediction_api.py        # 🌐  Flask REST API
│
├── ── Core ML Modules ──
│   ├── model_profiler.py    # GPU profiler (CUDA events)
│   ├── feature_extractor.py # Static feature extraction from nn.Module
│   ├── gnn_extractor.py     # GNN-based graph feature extraction
│   ├── gnn_model.py         # ArchitectureGNN model definition (PyG)
│   ├── prediction_model.py  # Train XGBoost / RF / Linear models
│   ├── train_gnn.py         # Train the GNN predictor
│   ├── train_memory_model.py# Train memory prediction model
│   ├── gpu_predictor.py     # Inference class with caching & batch support
│   ├── model_analyser.py    # Model complexity analysis utilities
│   ├── advanced_features.py # Extended feature engineering
│   ├── dynamic_predictor.py # Dynamic / online prediction
│   ├── gpu_info.py          # GPU metadata (pynvml)
│   ├── workload_scheduler.py# Batch workload scheduler
│   └── performance_monitor.py
│
├── scripts/                 # 🔬  Experiment & data scripts
│   ├── collect_data.py      # Profile CNN/Transformer/custom models → data/raw/
│   ├── enhance_dataset.py   # Augment dataset (more batch sizes / models)
│   ├── diverse_architectures.py  # Profile diverse arch families
│   ├── ablation_study.py    # 5-condition ablation (Table II in paper)
│   ├── generate_paper_figures.py # Reproduce all paper figures
│   └── generate_paper_tables.py  # Reproduce paper tables
│
├── tests/                   # ✅  Test suite
│   ├── test_diverse_models.py
│   ├── test_predictors.py
│   ├── test_profiler.py
│   ├── test_gnn_scaling.py
│   └── evaluate_gnn_vs_xgb.py
│
├── data/
│   ├── raw/                 # Raw profiling CSVs (gitignored)
│   ├── processed/           # Feature-engineered CSVs
│   ├── enriched/            # Final training-ready dataset
│   └── feedback_log.csv     # Online feedback loop log
│
├── models/                  # Serialized model artifacts (gitignored)
│   ├── xgboost_(tuned)_model.joblib
│   ├── random_forest_model.joblib
│   ├── gnn_predictor.pth
│   ├── memory_model.joblib
│   └── ...
│
├── results/
│   ├── figures/             # Paper figures (PNG)
│   ├── ablation_study_table.csv
│   ├── gnn_scaling_table.csv
│   └── ...
│
├── templates/index.html     # HTML template for web interface
├── legacy/                  # Archived / superseded scripts
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd neusight

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch Geometric (match your CUDA version)
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-geometric
```

**Requirements:** NVIDIA GPU with CUDA, Python ≥ 3.10

---

## Usage

### 1. Launch the Dashboard
```bash
streamlit run dashboard.py
```
Features: live model prediction, batch size optimizer, model comparison, performance monitor.

### 2. Collect Profiling Data
```bash
python scripts/collect_data.py --batch-sizes 1 4 16 32 64
```

### 3. Train Prediction Models
```bash
# Train XGBoost / RF / Linear baseline models
python prediction_model.py

# Train GNN predictor
python train_gnn.py

# Train memory model
python train_memory_model.py
```

### 4. Run Ablation Study
```bash
python scripts/ablation_study.py
```

### 5. Predict via Python API
```python
from gpu_predictor import GPUPredictor
import torchvision.models as models

predictor = GPUPredictor()
model = models.resnet50(pretrained=False)
result = predictor.predict_for_custom_model(model, batch_size=16)
print(result)
# {'execution_time_ms': 12.4, 'memory_mb': 1820, 'confidence_lower': 11.1, ...}
```

---

## Data Pipeline

```
collect_data.py
    └─▶ data/raw/*.csv          (GPU profiling measurements)
            │
            ▼
feature_extractor.py
    └─▶ data/processed/*.csv    (static model features)
            │
            ▼
enhance_dataset.py
    └─▶ data/enriched/*.csv     (augmented, training-ready)
            │
            ▼
prediction_model.py / train_gnn.py
    └─▶ models/                 (trained predictors)
```

---

## Model Performance

Results on held-out test set (20% split):

| Model | Exec Time MAPE | Memory MAPE | Notes |
|---|---|---|---|
| XGBoost (tuned) | ~8% | ~6% | Best overall |
| Random Forest | ~11% | ~9% | Robust baseline |
| GNN Predictor | ~10% | ~8% | Best on unseen architectures |
| Linear Regression | ~22% | ~19% | Baseline |

*(Full ablation study results: `results/ablation_study_table.csv`)*

---

## Dashboard

The Streamlit dashboard (`dashboard.py`) provides:

| Tab | Description |
|---|---|
| 🎯 Prediction | Predict execution time & memory for standard or custom models |
| ⚡ Batch Optimizer | Find optimal batch size within a memory budget |
| 📊 Model Comparison | Compare predictions across multiple architectures |
| 📈 Performance Monitor | Live GPU utilization and prediction history |

---

## Paper Reproducibility

To reproduce all paper figures and tables:
```bash
python scripts/generate_paper_figures.py
python scripts/generate_paper_tables.py
python scripts/ablation_study.py
```
Outputs saved to `results/figures/`.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
