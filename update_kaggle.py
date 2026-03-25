import json
with open('colab_data_collector.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

source = data['cells'][3]['source']
for i, line in enumerate(source):
    if '"NVIDIA A100"' in line:
        source.insert(i, '    "Tesla P100": {"tflops_fp32": 9.3, "memory_bandwidth_gbps": 732, "sm_count": 56},\n')
        source.insert(i+1, '    "Tesla P100-PCIE-16GB": {"tflops_fp32": 9.3, "memory_bandwidth_gbps": 732, "sm_count": 56},\n')
        break

data['cells'][3]['source'] = source

with open('colab_data_collector.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
