#!/usr/bin/env python3
"""
blink_github_action.py  —  CI/CD SLA Enforcement Script for Blink

Usage (standalone, in a PR check shell step):
    python blink_github_action.py \\
        --model-name  resnet50 \\
        --batch-size  32 \\
        --sla-latency-ms  50 \\
        --sla-memory-mb   4096

    Returns exit code 0 if the model passes the SLA.
    Returns exit code 1 if the model BREACHES the SLA (breaks the CI build).

For LLMs, also specify:
    --seq-len          128
    --quantization-bits 16
    --sla-prefill-ms   1000   (Time-To-First-Token budget)
    --sla-decode-ms    100    (Time-Per-Output-Token budget)
"""

import argparse
import sys
import json
import os


def print_header():
    print("=" * 64)
    print("  🔭 Blink CI/CD — GPU SLA Check")
    print("=" * 64)


def print_result(label, value, limit, unit, passed):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {label:30s} {value:>10.1f} {unit}  (limit: {limit} {unit})")


def run_sla_check(args):
    """Run the Blink predictor for the specified model and check against SLAs."""
    print_header()

    # ── Import predictor ──────────────────────────────────────────────────────
    try:
        # Add the project root to the path so blink modules can be found
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)

        from blink.gpu_predictor import GPUPredictor
        from blink.model_analyser import ModelAnalyzer
    except ImportError as e:
        print(f"\n[ERROR] Could not import Blink modules: {e}")
        print("Make sure you are running from the project root directory.")
        sys.exit(2)

    predictor = GPUPredictor()
    analyzer  = ModelAnalyzer()

    # ── Resolve model ──────────────────────────────────────────────────────────
    model_obj = None

    if args.model_name:
        # Try HuggingFace Causal LM first if seq_len is specified
        if args.seq_len and args.seq_len > 0:
            try:
                from transformers import AutoModelForCausalLM
                print(f"\n  Loading HuggingFace model: {args.model_name}")
                load_kwargs = {}
                if args.quantization_bits == 16:
                    import torch
                    load_kwargs["torch_dtype"] = torch.float16
                model_obj = AutoModelForCausalLM.from_pretrained(
                    args.model_name, **load_kwargs
                )
            except Exception as e:
                print(f"  Could not load as HF Causal LM ({e}), trying TorchVision...")

        if model_obj is None:
            # Try TorchVision model
            try:
                import torchvision.models as tv
                model_factory = getattr(tv, args.model_name, None)
                if model_factory:
                    model_obj = model_factory(weights=None)
                    print(f"\n  Loaded TorchVision model: {args.model_name}")
            except Exception as e:
                print(f"  [ERROR] Could not load model '{args.model_name}': {e}")
                sys.exit(2)

    if model_obj is None:
        print(f"\n[ERROR] Could not resolve model: {args.model_name}")
        sys.exit(2)

    # ── Extract features & predict ─────────────────────────────────────────────
    print(f"\n  Analysing model (batch_size={args.batch_size}, "
          f"seq_len={args.seq_len}, quant={args.quantization_bits}-bit)...")

    features = analyzer.extract_features(
        model_obj,
        batch_size=args.batch_size,
        seq_len=args.seq_len or 128,
        quantization_bits=args.quantization_bits,
    )
    is_llm = features.get("is_llm", False)

    # Use the predictor for latency/memory estimates
    pred = predictor.predict_for_custom_model(model_obj, args.batch_size)

    exec_upper_ms     = pred.get("exec_upper_ms",    pred.get("exec_time_ms", 0))
    memory_upper_mb   = pred.get("memory_upper_mb",  pred.get("memory_usage_mb", 0))
    prefill_memory_mb = features.get("prefill_memory_mb", memory_upper_mb)
    decode_memory_mb  = features.get("decode_memory_mb",  memory_upper_mb)

    # For LLMs, treat prefill_time ≈ exec_upper and use KV-cache-adjusted memory
    # (actual prefill/decode timing requires ground-truth from collect_data.py)
    print()
    print("  Prediction Results:")
    print("  " + "-" * 60)

    all_passed = True
    violations = []

    # ── Latency SLA ────────────────────────────────────────────────────────────
    if args.sla_latency_ms is not None:
        passed = exec_upper_ms <= args.sla_latency_ms
        print_result("Exec time (P90 upper bound)", exec_upper_ms,
                     args.sla_latency_ms, "ms", passed)
        if not passed:
            all_passed = False
            violations.append(
                f"Exec time {exec_upper_ms:.1f}ms exceeds SLA of {args.sla_latency_ms}ms "
                f"(+{exec_upper_ms - args.sla_latency_ms:.1f}ms over budget)"
            )

    # ── Prefill SLA (LLM-specific: Time-To-First-Token) ──────────────────────
    if args.sla_prefill_ms is not None and is_llm:
        passed = exec_upper_ms <= args.sla_prefill_ms
        print_result("Prefill / TTFT (P90 upper)", exec_upper_ms,
                     args.sla_prefill_ms, "ms", passed)
        if not passed:
            all_passed = False
            violations.append(
                f"Prefill (TTFT) {exec_upper_ms:.1f}ms exceeds SLA of {args.sla_prefill_ms}ms"
            )

    # ── Memory SLA ─────────────────────────────────────────────────────────────
    if args.sla_memory_mb is not None:
        mem_check = prefill_memory_mb if is_llm else memory_upper_mb
        passed = mem_check <= args.sla_memory_mb
        label  = "Peak memory (prefill)" if is_llm else "Peak memory (P90 upper)"
        print_result(label, mem_check, args.sla_memory_mb, "MB", passed)
        if not passed:
            all_passed = False
            violations.append(
                f"Peak memory {mem_check:.0f}MB exceeds SLA of {args.sla_memory_mb}MB "
                f"(+{mem_check - args.sla_memory_mb:.0f}MB over budget)"
            )

    if is_llm and args.sla_memory_mb is not None:
        # Also report decode memory (usually smaller, as no new activations)
        print_result("Decode memory estimate", decode_memory_mb,
                     args.sla_memory_mb, "MB", decode_memory_mb <= args.sla_memory_mb)

    print("  " + "-" * 60)

    # ── Summary ────────────────────────────────────────────────────────────────
    if all_passed:
        print("\n  🎉 All SLA checks PASSED. Build can proceed.\n")

        # Emit a GitHub Actions output (no-op outside of GH Actions)
        gha_output = os.environ.get("GITHUB_OUTPUT", "")
        if gha_output:
            with open(gha_output, "a") as f:
                f.write(f"blink_status=PASS\n")
                f.write(f"blink_exec_upper_ms={exec_upper_ms:.2f}\n")

        sys.exit(0)
    else:
        print("\n  ⛔ SLA VIOLATION — Build BLOCKED by Blink.\n")
        print("  Violations:")
        for v in violations:
            print(f"    • {v}")

        print()
        print("  💡 Suggested fixes:")
        print("    - Reduce model depth or channel width to lower latency.")
        print("    - Apply quantization (fp16 / int8) to reduce memory footprint.")
        print("    - Increase the SLA budget if this latency is acceptable.")

        # Emit failure to GitHub Actions
        gha_output = os.environ.get("GITHUB_OUTPUT", "")
        if gha_output:
            with open(gha_output, "a") as f:
                f.write(f"blink_status=FAIL\n")
                f.write(f"blink_violations={json.dumps(violations)}\n")

        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Blink CI/CD SLA Check — predict GPU cost and enforce latency/memory budgets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model specification
    parser.add_argument("--model-name", required=True,
                        help="TorchVision model name (e.g. resnet50) or HuggingFace model ID (e.g. gpt2)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to predict for (default: 1)")

    # LLM-specific
    parser.add_argument("--seq-len", type=int, default=0,
                        help="Token sequence length for LLMs (0 = treat as Vision model)")
    parser.add_argument("--quantization-bits", type=int, default=32, choices=[32, 16, 8, 4],
                        help="Model precision: 32 (fp32), 16 (fp16), 8 (int8), 4 (int4)")

    # SLA budgets
    parser.add_argument("--sla-latency-ms", type=float, default=None,
                        help="Maximum allowed P90 execution time in milliseconds")
    parser.add_argument("--sla-prefill-ms", type=float, default=None,
                        help="(LLM) Max Time-To-First-Token in milliseconds")
    parser.add_argument("--sla-decode-ms", type=float, default=None,
                        help="(LLM) Max Time-Per-Output-Token in milliseconds (informational)")
    parser.add_argument("--sla-memory-mb", type=float, default=None,
                        help="Maximum allowed peak GPU memory in MB")

    args = parser.parse_args()

    if args.sla_latency_ms is None and args.sla_memory_mb is None and args.sla_prefill_ms is None:
        print("[ERROR] At least one SLA must be specified "
              "(--sla-latency-ms, --sla-prefill-ms, or --sla-memory-mb).")
        sys.exit(2)

    run_sla_check(args)


if __name__ == "__main__":
    main()
