import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

os.makedirs('figures', exist_ok=True)

# Set style for IEEE paper
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="whitegrid")

# 1. Batch Scaling Plot (Table I)
models = ['ResNet18', 'ResNet50', 'VGG16', 'MobileNetV2', 'EfficientNet', 'DenseNet121', 'ShuffleNetV2', 'ConvNeXt-Tiny']
batch_sizes = [1, 4, 16, 32, 64]
data = [
    [1.90, 4.57, 10.37, 19.14, 36.73],
    [4.33, 12.10, 31.42, 60.30, 120.89],
    [6.96, 30.95, 71.22, 217.50, 840.65],
    [4.83, 9.89, 10.87, 23.14, 45.26],
    [6.93, 6.83, 18.54, 50.12, 145.93],
    [12.56, 33.60, 33.45, 99.46, 177.49],
    [7.89, 7.19, 6.85, 24.86, 45.93],
    [16.74, 60.16, 249.40, 494.43, 865.83]
]

plt.figure(figsize=(7, 4.5))
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
for i, (model, times) in enumerate(zip(models, data)):
    plt.plot(batch_sizes, times, marker=markers[i], label=model, linewidth=2, markersize=6)

plt.xlabel('Batch Size')
plt.ylabel('Measured Execution Time (ms)')
plt.yscale('log')
plt.xscale('log', base=2)
plt.xticks(batch_sizes, [str(b) for b in batch_sizes])
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('GPU Execution Time Scaling with Batch Size (RTX 3060)')
plt.tight_layout()
plt.savefig('figures/batch_scaling.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Algorithm Comparison (Table II)
algos = ['Elastic Net', 'SVR (RBF Kernel)', 'Random Forest', 'XGBoost (Static)', 'Blink (Ours)']
latency_mape = [22.4, 18.6, 11.2, 9.8, 7.8]
memory_mape = [19.3, 16.5, 9.4, 8.1, 6.2]

x = np.arange(len(algos))
width = 0.35

plt.figure(figsize=(7, 4))
fig, ax = plt.subplots(figsize=(7, 4))
rects1 = ax.bar(x - width/2, latency_mape, width, label='Execution Latency', color='#1f77b4', edgecolor='black')
rects2 = ax.bar(x + width/2, memory_mape, width, label='Peak GPU Memory', color='#ff7f0e', edgecolor='black')

ax.set_ylabel('Mean Absolute Percentage Error (MAPE, %)')
ax.set_title('Prediction Accuracy by Algorithm')
ax.set_xticks(x)
ax.set_xticklabels(algos, rotation=15, ha='right')
ax.legend()

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('figures/algo_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Ablation Study Chart (Table III)
configs = ['Static Only', 'Static + GNN', 'Static + Optuna', 'Blink (Static + GNN + Optuna)']
lat_ablation = [9.8, 8.6, 8.9, 7.8]
mem_ablation = [8.1, 7.2, 7.5, 6.2]

x_ablation = np.arange(len(configs))

plt.figure(figsize=(7, 4))
fig, ax = plt.subplots(figsize=(7, 4))
rects1_ab = ax.bar(x_ablation - width/2, lat_ablation, width, label='Latency MAPE', color='#2ca02c', edgecolor='black')
rects2_ab = ax.bar(x_ablation + width/2, mem_ablation, width, label='Memory MAPE', color='#d62728', edgecolor='black')

ax.set_ylabel('Mean Absolute Percentage Error (MAPE, %)')
ax.set_title('Ablation Study: GNN and Optuna Contributions')
ax.set_xticks(x_ablation)
ax.set_xticklabels(configs, rotation=10, ha='right')
ax.legend()
autolabel(rects1_ab)
autolabel(rects2_ab)

plt.tight_layout()
plt.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight')
plt.close()

print('Figures generated successfully!')
