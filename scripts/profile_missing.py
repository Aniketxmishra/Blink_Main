"""
Profile all torchvision models NOT yet in data/raw/, saving one CSV per model.
Runs on CPU if no GPU is present so it works on any machine (just slower).
"""
import torch
import torchvision.models as tv_models
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZES    = [1, 2, 4, 8, 16, 32, 64, 128]
NUM_WARMUP     = 3
NUM_TIMED      = 5
SAVE_DIR       = "data/raw"
INPUT_SIZE     = (3, 224, 224)

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# GPU hardware specs (RTX 3060 defaults; will be filled properly when CUDA available)
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_name = name
else:
    gpu_name = "CPU"

# ── Find already-profiled architectures ──────────────────────────────────────
existing = set()
for f in os.listdir(SAVE_DIR):
    if f.endswith(".csv") and "Tesla" not in f:
        # strip timestamp suffix from filename: model_YYYYMMDD_HHMMSS.csv
        parts = f.replace(".csv", "").split("_")
        # Last two parts are date and time; everything before is the model name
        model_name = "_".join(parts[:-2])
        existing.add(model_name)

print(f"Already profiled ({len(existing)}): {sorted(existing)}")

# ── All torchvision constructors ──────────────────────────────────────────────
SKIP = {"get_model", "get_model_builder", "get_model_weights", "get_weight", "list_models"}
all_tv = sorted([
    n for n in dir(tv_models)
    if callable(getattr(tv_models, n))
    and not n.startswith("_")
    and n[0].islower()
    and n not in SKIP
])

missing = [m for m in all_tv if m not in existing]
print(f"\nMissing {len(missing)} models to profile: {missing}")

# ── Profile helper ────────────────────────────────────────────────────────────
def profile_model(model_name, model, input_shape):
    rows = []
    model = model.to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buf_bytes    = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_mb     = (param_bytes + buf_bytes) / (1024 ** 2)

    for bs in BATCH_SIZES:
        try:
            x = torch.zeros(bs, *input_shape, device=device)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                # Warmup
                with torch.no_grad():
                    for _ in range(NUM_WARMUP):
                        model(x)
                torch.cuda.synchronize()

                start_evt = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMED)]
                end_evt   = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMED)]
                with torch.no_grad():
                    for i in range(NUM_TIMED):
                        start_evt[i].record()
                        model(x)
                        end_evt[i].record()
                torch.cuda.synchronize()
                times_ms = [s.elapsed_time(e) for s, e in zip(start_evt, end_evt)]
                peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            else:
                # CPU timing
                with torch.no_grad():
                    for _ in range(NUM_WARMUP):
                        model(x)
                times_ms = []
                with torch.no_grad():
                    for _ in range(NUM_TIMED):
                        t0 = time.perf_counter()
                        model(x)
                        times_ms.append((time.perf_counter() - t0) * 1000)
                peak_mem_mb = 0.0

            exec_ms = float(np.median(times_ms))
            std_ms  = float(np.std(times_ms))
            cv      = std_ms / exec_ms if exec_ms > 0 else 0.0

            rows.append({
                "model_name":           model_name,
                "batch_size":           bs,
                "input_shape":          str((bs,) + input_shape),
                "total_parameters":     total_params,
                "trainable_parameters": trainable,
                "model_size_mb":        model_mb,
                "device":               str(device),
                "execution_time_ms":    exec_ms,
                "execution_time_std":   std_ms,
                "timing_cv":            cv,
                "peak_memory_mb":       peak_mem_mb,
                "gpu_name":             gpu_name,
                # Hardware specs
                "tflops_fp32":          12.0 if device.type == "cuda" else 0.0,
                "memory_bandwidth_gbps": 336.0 if device.type == "cuda" else 0.0,
                "sm_count":             28 if device.type == "cuda" else 0,
            })
            print(f"  bs={bs:3d} → {exec_ms:8.2f} ms  peak={peak_mem_mb:.0f} MB")

        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs}: OOM — stopping")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  bs={bs}: error — {e}")
            break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows

# ── Main loop ─────────────────────────────────────────────────────────────────
total_new = 0
for model_name in missing:
    print(f"\n{'='*60}")
    print(f"Profiling: {model_name}")
    try:
        model_fn = getattr(tv_models, model_name)
        input_shape = (3, 299, 299) if "inception" in model_name else (3, 224, 224)
        model = model_fn(weights=None)
    except Exception as e:
        print(f"  Cannot instantiate {model_name}: {e} — skipping")
        continue

    rows = profile_model(model_name, model, input_shape)
    if rows:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SAVE_DIR, f"{model_name}_{ts}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  ✓ Saved {len(rows)} rows → {path}")
        total_new += 1
    else:
        print(f"  ✗ No rows collected for {model_name}")

print(f"\n{'='*60}")
print(f"Done! Profiled {total_new} new architectures.")
print("Now run: python -m scripts.enhance_dataset   to re-enrich the full dataset.")
