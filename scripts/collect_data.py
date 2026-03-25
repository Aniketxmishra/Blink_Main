from model_profiler import ModelProfiler
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
import time
import pandas as pd
from datetime import datetime


NUM_WARMUP_RUNS = 3   # warm up GPU caches before timing
NUM_TIMING_RUNS = 5   # number of timed runs to average



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect GPU usage data for various models')
    parser.add_argument('--batch-sizes', type=int, nargs='+', 
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                        help='Batch sizes to profile')
    parser.add_argument('--save-dir', type=str, default='data/enriched',
                        help='Directory to save profiling results')

    parser.add_argument('--model-type', type=str, default='all',
                        choices=['all', 'cnn', 'transformer', 'llm', 'custom'],
                        help='Type of models to profile')
    parser.add_argument('--num-runs', type=int, default=NUM_TIMING_RUNS,
                        help='Number of profiling runs per config (results averaged)')
    parser.add_argument('--seq-lengths', type=int, nargs='+',
                        default=[64, 128, 256, 512],
                        help='Token sequence lengths to sweep for LLMs')
    args = parser.parse_args()
    
    # Create a model profiler
    profiler = ModelProfiler(save_dir=args.save_dir)
    
    # Profile CNN models
    if args.model_type in ['all', 'cnn']:
        print("Profiling CNN models...")
        profile_cnn_models(profiler, args.batch_sizes)
    
    # Profile vision transformer models
    if args.model_type in ['all', 'transformer']:
        print("Profiling vision transformer models...")
        profile_transformer_models(profiler, args.batch_sizes)

    # Profile Causal LLMs with prefill/decode separation
    if args.model_type in ['all', 'llm']:
        print("Profiling Causal LLM models (prefill + decode)...")
        profile_causal_llms(profiler, args.batch_sizes, args.seq_lengths, args.save_dir)
    
    # Profile custom models
    if args.model_type in ['all', 'custom']:
        print("Profiling custom models...")
        profile_custom_models(profiler, args.batch_sizes)
    
    print("Data collection complete!")


def profile_cnn_models(profiler, batch_sizes):
    """Profile common CNN architectures"""
    cnn_models = {
        # Vision Transformers (TorchVision)
        "vit_b_16":  models.vit_b_16(weights=None),
        "vit_b_32":  models.vit_b_32(weights=None),
        "vit_l_16":  models.vit_l_16(weights=None),
        "swin_t":    models.swin_t(weights=None),
        "swin_s":    models.swin_s(weights=None),
        "swin_b":    models.swin_b(weights=None),
        "swin_v2_t": models.swin_v2_t(weights=None),
        "swin_v2_s": models.swin_v2_s(weights=None),

        # MaxViT (hybrid CNN+Transformer)
        "maxvit_t": models.maxvit_t(weights=None),
    }
    
    all_results = []
    
    for name, model in cnn_models.items():
        print(f"Profiling {name}...")
        results = profiler.profile_model(
            model=model,
            input_shape=(3, 224, 224),
            batch_sizes=batch_sizes,
            model_name=name
        )
        all_results.append(results)
    
    return all_results

def profile_transformer_models(profiler, batch_sizes):
    """Profile encoder-style HuggingFace transformer models (BERT, RoBERTa)."""
    try:
        from transformers import BertModel, RobertaModel
        
        transformer_models = {}
        
        try:
            transformer_models["bert-base"] = BertModel.from_pretrained("bert-base-uncased")
        except Exception as e:
            print(f"Could not load BERT: {e}")
        
        try:
            transformer_models["roberta-base"] = RobertaModel.from_pretrained("roberta-base")
        except Exception as e:
            print(f"Could not load RoBERTa: {e}")
        
        all_results = []
        for name, model in transformer_models.items():
            print(f"Profiling {name}...")
            results = _profile_encoder_model(
                profiler=profiler, model=model,
                batch_sizes=batch_sizes, model_name=name
            )
            all_results.append(results)
        
        return all_results
    
    except ImportError:
        print("Transformers library not installed. Skipping transformer models.")
        return []


def _profile_encoder_model(profiler, model, batch_sizes, model_name):
    """Profile an encoder-style transformer (BERT/RoBERTa) with token input."""
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_params   = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size     = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size    = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb  = (param_size + buffer_size) / (1024 ** 2)

    for batch_size in batch_sizes:
        try:
            input_ids      = torch.randint(0, 30000, (batch_size, 128), device=device)
            attention_mask = torch.ones((batch_size, 128), device=device, dtype=torch.long)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

            times = []
            with torch.no_grad():
                for _ in range(NUM_WARMUP_RUNS):
                    model(input_ids=input_ids, attention_mask=attention_mask)
                for _ in range(NUM_TIMING_RUNS):
                    t0 = time.perf_counter()
                    model(input_ids=input_ids, attention_mask=attention_mask)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    times.append((time.perf_counter() - t0) * 1000)

            exec_ms = float(sum(times) / len(times))
            results.append({
                "model_name":          model_name,
                "batch_size":          batch_size,
                "execution_time_ms":   exec_ms,
                "total_parameters":    total_params,
                "trainable_parameters":trainable_params,
                "model_size_mb":       model_size_mb,
            })
            print(f"  {model_name} bs={batch_size}: {exec_ms:.2f} ms")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  Error {model_name} bs={batch_size}: {e}")
            break

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────

CAUSAL_LLM_CONFIGS = [
    # (hf_model_id,         friendly_name,    fp16,  int8,  int4)
    ("gpt2",                "gpt2",           True,  True,  False),
    ("gpt2-medium",         "gpt2-medium",    True,  True,  False),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                            "tinyllama-1.1b", True,  True,  True),
]


def profile_causal_llms(profiler, batch_sizes, seq_lengths, save_dir):
    """Profile Causal LLMs with separated Prefill (TTFT) and Decode (TPOT) timings.

    Also sweeps quantization: fp32, fp16, int8 (bitsandbytes), int4 (bitsandbytes).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers not installed. Skipping LLM profiling.")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []

    for model_id, friendly_name, do_fp16, do_int8, do_int4 in CAUSAL_LLM_CONFIGS:
        configs_to_run = [(32, {})]   # fp32 baseline
        if do_fp16:
            configs_to_run.append((16, {"torch_dtype": torch.float16}))
        if do_int8:
            try:
                import bitsandbytes  # noqa: F401
                configs_to_run.append((8, {"load_in_8bit": True}))
            except ImportError:
                print("bitsandbytes not installed — skipping int8 configs.")
        if do_int4:
            try:
                import bitsandbytes  # noqa: F401
                configs_to_run.append((4, {"load_in_4bit": True}))
            except ImportError:
                pass

        for quant_bits, load_kwargs in configs_to_run:
            label = f"{friendly_name}_int{quant_bits}" if quant_bits < 32 else f"{friendly_name}_fp32"
            print(f"  Loading {label}...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto" if torch.cuda.is_available() else None,
                    **load_kwargs
                )
                model.eval()
            except Exception as e:
                print(f"  Could not load {label}: {e}")
                continue

            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    row = _profile_prefill_and_decode(
                        model=model, model_name=label,
                        batch_size=batch_size, seq_len=seq_len,
                        quantization_bits=quant_bits, device=device
                    )
                    if row:
                        all_rows.append(row)
                        print(
                            f"    bs={batch_size} seq={seq_len} "
                            f"prefill={row.get('prefill_time_ms', 'N/A'):.1f}ms "
                            f"decode={row.get('decode_time_ms', 'N/A'):.2f}ms "
                            f"mem={row.get('peak_memory_mb', 'N/A'):.0f}MB"
                        )

            # Free model memory between configs
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if all_rows:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(save_dir, f"causal_llm_{ts}.csv")
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f"  Saved LLM profiling data to {out_path}")

    return all_rows


def _profile_prefill_and_decode(model, model_name, batch_size, seq_len,
                                 quantization_bits, device):
    """Time BOTH the prefill phase and the single-token decode phase.

    Returns:
        dict with prefill_time_ms, decode_time_ms, peak_memory_mb, and metadata.
        Returns None on OOM or other fatal errors.

    Prefill (Time-to-First-Token):
        Feed the FULL prompt (batch_size × seq_len) in one forward pass.
        This is the latency the user experiences before seeing the first token.

    Decode (Time-per-Output-Token):
        Feed a SINGLE new token while the KV cache for the full prompt is held in memory.
        This is the latency per generated token in the autoregressive decode loop.
    """
    try:
        # Build dummy tokenized inputs
        input_ids      = torch.randint(0, 1000, (batch_size, seq_len),   device=device)
        single_tok_ids = torch.randint(0, 1000, (batch_size, 1),         device=device)
        attn_full  = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        attn_single = torch.ones((batch_size, 1), device=device, dtype=torch.long)

        # ── Prefill ─────────────────────────────────────────────────────────
        # Run the model on the full sequence with use_cache=True so it
        # fills the KV cache (exactly mimicking the prefill stage).
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        with torch.no_grad():
            # Warmup
            for _ in range(NUM_WARMUP_RUNS):
                _ = model(input_ids=input_ids, attention_mask=attn_full,
                           use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

            # Timed runs
            prefill_times = []
            for _ in range(NUM_TIMING_RUNS):
                t0 = time.perf_counter()
                out = model(input_ids=input_ids, attention_mask=attn_full,
                            use_cache=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                prefill_times.append((time.perf_counter() - t0) * 1000)
            past_kv = out.past_key_values   # hold KV cache for decode step

        prefill_time_ms = float(sum(prefill_times) / len(prefill_times))
        peak_memory_mb  = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # ── Decode (single-token, KV-cache attended) ─────────────────────────
        # Extend the attention mask by 1 to cover the cached tokens + new token
        attn_extended = torch.ones((batch_size, seq_len + 1), device=device, dtype=torch.long)

        with torch.no_grad():
            # Warmup
            for _ in range(NUM_WARMUP_RUNS):
                _ = model(input_ids=single_tok_ids, attention_mask=attn_extended,
                           past_key_values=past_kv, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

            decode_times = []
            for _ in range(NUM_TIMING_RUNS):
                t0 = time.perf_counter()
                _ = model(input_ids=single_tok_ids, attention_mask=attn_extended,
                          past_key_values=past_kv, use_cache=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                decode_times.append((time.perf_counter() - t0) * 1000)

        decode_time_ms = float(sum(decode_times) / len(decode_times))
        # Throughput: tokens per second for the decode phase
        decode_tps     = (batch_size * 1000.0) / max(decode_time_ms, 0.001)

        return {
            "model_name":        model_name,
            "batch_size":        batch_size,
            "seq_len":           seq_len,
            "quantization_bits": quantization_bits,
            "prefill_time_ms":   prefill_time_ms,
            "decode_time_ms":    decode_time_ms,
            "decode_tps":        decode_tps,
            "peak_memory_mb":    peak_memory_mb,
            # Keep execution_time_ms = prefill for back-compat with existing predictor
            "execution_time_ms": prefill_time_ms,
            "is_llm":            True,
        }

    except torch.cuda.OutOfMemoryError:
        print(f"    OOM: bs={batch_size} seq={seq_len} on {model_name}")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"    Error profiling {model_name} bs={batch_size} seq={seq_len}: {e}")
        return None



def profile_custom_models(profiler, batch_sizes):
    """Profile custom model architectures"""
    
    class SimpleCNN(nn.Module):
        def __init__(self, num_layers=3, channels=16):
            super(SimpleCNN, self).__init__()
            layers = []
            in_channels = 3
            
            for i in range(num_layers):
                out_channels = channels * (2 ** i)
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2))
                in_channels = out_channels
            
            self.features = nn.Sequential(*layers)
            self.classifier = nn.Linear(out_channels * (224 // (2**num_layers)) * (224 // (2**num_layers)), 10)
            
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    custom_models = {
        "simple_cnn_3layers": SimpleCNN(num_layers=3, channels=16),
        "simple_cnn_5layers": SimpleCNN(num_layers=5, channels=16),
        "simple_cnn_3layers_wide": SimpleCNN(num_layers=3, channels=32),
    }
    
    all_results = []
    
    for name, model in custom_models.items():
        print(f"Profiling {name}...")
        results = profiler.profile_model(
            model=model,
            input_shape=(3, 224, 224),
            batch_sizes=batch_sizes,
            model_name=name
        )
        all_results.append(results)
    
    return all_results

if __name__ == "__main__":
    main()
