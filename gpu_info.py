import pynvml as nvml

# ── Quantization-Aware Hardware Spec Table ────────────────────────────────────
# Maps GPU model name substrings to their theoretical peak performance.
# tflops values are given for fp32 as the baseline; tensor-core precisions
# (fp16, int8, int4) are derived via the standard multipliers below.
#
# Standard Nvidia multipliers over fp32:
#   fp16  → 2×    (standard tensor core speedup)
#   int8  → 4×    (INT8 tensor cores)
#   int4  → 8×    (INT4 sparsity tensor cores, Ampere+)
#
# Memory bandwidth is precision-independent (it's a physical bus limit).

GPU_SPECS = {
    # key substring : (tflops_fp32, memory_bandwidth_gbps, sm_count, total_memory_gb)
    "H100":     (67.0,  3350.0, 132, 80.0),
    "A100":     (19.5,  2039.0,  108, 80.0),
    "A10G":     (31.2,   600.0,   72, 24.0),
    "A6000":    (38.7,   768.0,   84, 48.0),
    "RTX 4090": (82.6,  1008.0,  128, 24.0),
    "RTX 3090": (35.6,   936.0,   82, 24.0),
    "RTX 3080": (29.8,   760.0,   68, 10.0),
    "RTX 3070": (20.3,   448.0,   46,  8.0),
    "RTX 3060": (12.7,   360.0,   28, 12.0),
    "L4":       (30.3,   300.0,   58, 24.0),
    "T4":       ( 8.1,   300.0,   40, 16.0),
    "V100":     (14.0,   900.0,   80, 32.0),
    "P100":     ( 9.3,   732.0,   56, 16.0),
}

# Tensor-core speedup multipliers relative to fp32 baseline
QUANTIZATION_TFLOPS_MULTIPLIER = {
    32: 1.0,   # fp32  — baseline
    16: 2.0,   # fp16  — standard tensor cores
     8: 4.0,   # int8  — INT8 tensor cores
     4: 8.0,   # int4  — INT4 sparsity tensor cores (Ampere+)
}


def get_hardware_specs(gpu_name: str, quantization_bits: int = 32) -> dict:
    """Return the hardware performance specs for a given GPU name and precision.

    Args:
        gpu_name: Full GPU name string (e.g. "NVIDIA A100-SXM4-80GB").
        quantization_bits: Target precision (32, 16, 8, or 4).

    Returns:
        dict with tflops_fp32 (at target precision), memory_bandwidth_gbps,
        sm_count, total_memory_mb, and the effective_tflops.
    """
    # Find matching spec
    matched = None
    for key, specs in GPU_SPECS.items():
        if key.upper() in gpu_name.upper():
            matched = specs
            break

    if matched is None:
        # Fallback: RTX 3060-class defaults
        matched = (12.7, 360.0, 28, 12.0)

    tflops_fp32, bw_gbps, sm_count, mem_gb = matched
    multiplier   = QUANTIZATION_TFLOPS_MULTIPLIER.get(quantization_bits, 1.0)
    effective_tflops = tflops_fp32 * multiplier

    return {
        "tflops_fp32":              tflops_fp32,
        "effective_tflops":         effective_tflops,
        "memory_bandwidth_gbps":    bw_gbps,
        "sm_count":                 sm_count,
        "total_memory_mb":          mem_gb * 1024,
        "quantization_bits":        quantization_bits,
        "tflops_multiplier":        multiplier,
    }


def get_gpu_info(quantization_bits: int = 32) -> dict:
    """Get basic information about the available GPU, with quantization-aware TFLOPS.

    Args:
        quantization_bits: Target inference precision (32, 16, 8, or 4).
    """
    try:
        nvml.nvmlInit()
    except Exception:
        return {"error": "NVML not available — is pynvml installed and a GPU present?"}

    try:
        device_count = nvml.nvmlDeviceGetCount()
        if device_count == 0:
            return {"error": "No GPU found"}
    except nvml.NVMLError:
        return {"error": "NVML initialization error"}

    handle = nvml.nvmlDeviceGetHandleByIndex(0)

    # Get device name
    name = nvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode('utf-8')

    # Get memory info
    memory_info    = nvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory   = memory_info.total / (1024 ** 2)   # MB
    free_memory    = memory_info.free  / (1024 ** 2)   # MB

    # Get compute capability
    compute_capability = "Unknown"
    try:
        major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_capability = f"{major}.{minor}"
    except (AttributeError, nvml.NVMLError):
        if "K80" in name:
            compute_capability = "3.7"
        elif "P100" in name:
            compute_capability = "6.0"
        elif "V100" in name:
            compute_capability = "7.0"
        elif "T4" in name:
            compute_capability = "7.5"
        elif "A100" in name:
            compute_capability = "8.0"
        elif "H100" in name:
            compute_capability = "9.0"

    # Get clock speeds
    try:
        sm_clock  = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
        mem_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
    except nvml.NVMLError:
        sm_clock = mem_clock = 0

    nvml.nvmlShutdown()

    # Augment with quantization-aware hardware performance specs
    hw_specs = get_hardware_specs(name, quantization_bits)

    return {
        "name":                  name,
        "total_memory_mb":       total_memory,
        "free_memory_mb":        free_memory,
        "compute_capability":    compute_capability,
        "sm_clock_mhz":          sm_clock,
        "mem_clock_mhz":         mem_clock,
        **hw_specs,   # includes tflops_fp32, effective_tflops, memory_bandwidth_gbps, sm_count
    }


if __name__ == "__main__":
    for bits in [32, 16, 8, 4]:
        print(f"\n--- GPU Info @ int{bits} / fp{bits} precision ---")
        info = get_gpu_info(quantization_bits=bits)
        for key, value in info.items():
            print(f"  {key}: {value}")

