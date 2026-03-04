# ────────────────────────────────────────────────────────────────────────────
# Blink — Dockerfile
# Builds a single image that can run either:
#   - The Streamlit dashboard  (default, port 8501)
#   - The FastAPI REST server  (port 8000)
# Usage controlled by CMD in docker-compose.yml
# ────────────────────────────────────────────────────────────────────────────

# Use the official CUDA + Python base so pynvml can talk to the host GPU
# via nvidia-docker / the --gpus flag. Falls back gracefully without GPU.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── System packages ──────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3      1

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 wheels
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 \
        --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip install --no-cache-dir \
        torch-geometric \
        -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

RUN pip install --no-cache-dir \
        numpy==1.26.4 pandas==2.2.0 scikit-learn==1.4.0 \
        xgboost optuna lightgbm \
        thop==0.1.1.post2209072238 joblib==1.3.2 \
        pynvml \
        streamlit==1.32.0 plotly==5.18.0 matplotlib==3.8.3 seaborn==0.13.2 \
        "fastapi>=0.110.0" "uvicorn[standard]>=0.27.0" \
        python-multipart httpx

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# ── Expose ports ──────────────────────────────────────────────────────────────
EXPOSE 8501 8000

# ── Streamlit configuration ──────────────────────────────────────────────────
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# ── Default command: run the dashboard ───────────────────────────────────────
# Override in docker-compose.yml to run the API instead
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
