import pynvml as nvml

def get_gpu_info():
    """Get basic information about the available GPU"""
    nvml.nvmlInit()
    
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
    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = memory_info.total / (1024**2)  # Convert to MB
    
    # Get compute capability
    compute_capability = "Unknown"
    try:
        major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_capability = f"{major}.{minor}"
    except (AttributeError, nvml.NVMLError):
        # Infer from name
        if "K80" in name: compute_capability = "3.7"
        elif "P100" in name: compute_capability = "6.0"
        elif "V100" in name: compute_capability = "7.0"
        elif "T4" in name: compute_capability = "7.5"
        elif "A100" in name: compute_capability = "8.0"
        elif "H100" in name: compute_capability = "9.0"
    
    # Get clock speeds
    sm_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
    mem_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
    
    nvml.nvmlShutdown()
    
    return {
        "name": name,
        "total_memory_mb": total_memory,
        "compute_capability": compute_capability,
        "sm_clock_mhz": sm_clock,
        "mem_clock_mhz": mem_clock
    }

if __name__ == "__main__":
    # Test the function
    gpu_info = get_gpu_info()
    print("GPU Information:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
