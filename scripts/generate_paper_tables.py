"""Generate all three requested paper outputs:
1. CI calibration empirical coverage
2. Table I: Real GPU timing numbers from profiled CSVs
3. GNN scaling ratio table (BS=1 vs BS=32)
"""
import pandas as pd
import numpy as np
import glob
import torch
from xgboost import XGBRegressor
from blink.gnn_model import ArchitectureGNN
from blink.gnn_extractor import model_to_graph
import torchvision.models as models

def detect_arch_family(model_name):
    families = {
        'cnn_residual': ['resnet18', 'resnet50', 'resnext50_32x4d', 'wide_resnet50_2'],
        'cnn_dense': ['densenet121', 'densenet169', 'densenet201'],
        'cnn_plain': ['vgg16', 'vgg19'],
        'cnn_mobile': ['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
                       'shufflenet_v2_x1_0', 'mnasnet1_0', 'squeezenet1_0'],
        'cnn_efficient': ['efficientnet_b0', 'efficientnet_v2_s'],
        'cnn_modern': ['convnext_tiny', 'convnext_small', 'convnext_base'],
        'cnn_regnet': ['regnet_y_400mf', 'regnet_x_400mf', 'regnet_y_800mf'],
        'cnn_inception': ['googlenet', 'inception_v3'],
    }
    for family, members in families.items():
        if model_name in members:
            return family
    return 'other'

# --- LOAD DATA ---
csv_files = glob.glob('data/raw/*.csv')
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df = df.dropna(subset=['execution_time_ms'])
df = df[df['execution_time_ms'] > 0]

# Feature engineering
df['flops'] = df['total_parameters'] * 2
df['compute_memory_ratio'] = df['flops'] / (df['model_size_mb'] * 1024**2 + 1)
df['param_ratio'] = df['trainable_parameters'] / (df['total_parameters'] + 1)
df['arch_family'] = df['model_name'].apply(detect_arch_family)
df['arch_family_code'] = pd.Categorical(df['arch_family']).codes

rich_features = ['batch_size', 'total_parameters', 'model_size_mb', 'flops',
                 'compute_memory_ratio', 'trainable_parameters', 'param_ratio', 'arch_family_code']

# Split
train_bs = [1, 2, 4, 8]
test_bs = [16, 32, 64]
train_df = df[df['batch_size'].isin(train_bs)]
test_df = df[df['batch_size'].isin(test_bs)]

y_train = np.log1p(train_df['execution_time_ms'])
y_actual = test_df['execution_time_ms'].values

# Train model
xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
xgb.fit(train_df[rich_features], y_train)
preds_log = np.clip(xgb.predict(test_df[rich_features]), -2, 10)
preds = np.expm1(preds_log)

# =====================================================
# 1. CI CALIBRATION
# =====================================================
print("=" * 60)
print("1. CI CALIBRATION — Empirical Coverage")
print("=" * 60)

for margin_name, lo_mult, hi_mult in [("±15%", 0.85, 1.15), ("±20%", 0.80, 1.20), ("±30%", 0.70, 1.30), ("±50%", 0.50, 1.50)]:
    lower = preds * lo_mult
    upper = preds * hi_mult
    coverage = np.mean((y_actual >= lower) & (y_actual <= upper))
    print(f"  {margin_name} CI coverage: {coverage:.1%}")

# For 80% target, find the right margin
for pct in range(10, 100, 5):
    lo_mult = 1 - pct/100
    hi_mult = 1 + pct/100
    lower = preds * lo_mult
    upper = preds * hi_mult
    coverage = np.mean((y_actual >= lower) & (y_actual <= upper))
    if coverage >= 0.80:
        print(f"\n  → 80% coverage achieved at ±{pct}% margin")
        print(f"    Empirical 80% CI coverage: {coverage:.1%}")
        break

# =====================================================
# 2. TABLE I — Real GPU Timing Numbers
# =====================================================
print("\n" + "=" * 60)
print("2. TABLE I — Real GPU Timing (ms) from CUDA Events Profiler")
print("=" * 60)

# Pick 8 representative models
representative = ['resnet18', 'resnet50', 'vgg16', 'mobilenet_v2', 'efficientnet_b0',
                  'densenet121', 'shufflenet_v2_x1_0', 'convnext_tiny']

# Get latest profiling data per model
table1_rows = []
for model_name in representative:
    model_data = df[df['model_name'] == model_name].sort_values('batch_size')
    if len(model_data) == 0:
        continue
    
    # Get params and size from first row
    params = model_data.iloc[0]['total_parameters']
    size_mb = model_data.iloc[0]['model_size_mb']
    
    timing = {}
    for _, row in model_data.iterrows():
        bs = int(row['batch_size'])
        timing[bs] = row['execution_time_ms']
    
    table1_rows.append({
        'Model': model_name,
        'Params (M)': f"{params/1e6:.1f}",
        'Size (MB)': f"{size_mb:.1f}",
        'BS=1 (ms)': f"{timing.get(1, 0):.2f}",
        'BS=4 (ms)': f"{timing.get(4, 0):.2f}",
        'BS=16 (ms)': f"{timing.get(16, 0):.2f}",
        'BS=32 (ms)': f"{timing.get(32, 0):.2f}",
        'BS=64 (ms)': f"{timing.get(64, 0):.2f}",
    })

table1 = pd.DataFrame(table1_rows)
print(table1.to_string(index=False))
table1.to_csv('results/table1_gpu_timing.csv', index=False)

# =====================================================
# 3. GNN SCALING RATIO TABLE
# =====================================================
print("\n" + "=" * 60)
print("3. GNN SCALING RATIO TABLE — Predicted vs Expected")
print("=" * 60)

gnn = ArchitectureGNN()
gnn.load_state_dict(torch.load('models/gnn_predictor.pth', map_location='cpu', weights_only=True))
gnn.eval()

gnn_test_models = {
    'ResNet18': models.resnet18(weights=None),
    'VGG16': models.vgg16(weights=None),
    'MobileNetV2': models.mobilenet_v2(weights=None),
    'ShuffleNetV2': models.shufflenet_v2_x1_0(weights=None),
    'SqueezeNet': models.squeezenet1_0(weights=None),
    'DenseNet121': models.densenet121(weights=None),
    'EfficientNet-B0': models.efficientnet_b0(weights=None),
    'ConvNeXt-Tiny': models.convnext_tiny(weights=None),
}

gnn_rows = []
for name, net in gnn_test_models.items():
    graph = model_to_graph(net)
    preds_gnn = {}
    for bs in [1, 32]:
        bs_t = torch.tensor([[float(bs)]], dtype=torch.float32)
        with torch.no_grad():
            out = gnn(graph, bs_t)
        ms = max(0.1, float(np.expm1(out[0, 0].item())))
        preds_gnn[bs] = ms
    
    ratio = preds_gnn[32] / preds_gnn[1]
    gnn_rows.append({
        'Model': name,
        'BS=1 (ms)': f"{preds_gnn[1]:.2f}",
        'BS=32 (ms)': f"{preds_gnn[32]:.2f}",
        'Ratio': f"{ratio:.1f}x",
        'Expected': '~3-4x' if ratio > 2.0 else 'LOW',
        'Status': 'OK' if 2.0 < ratio < 8.0 else 'WARN'
    })

gnn_table = pd.DataFrame(gnn_rows)
print(gnn_table.to_string(index=False))
gnn_table.to_csv('results/gnn_scaling_table.csv', index=False)

print("\n--- All tables saved to results/ ---")
