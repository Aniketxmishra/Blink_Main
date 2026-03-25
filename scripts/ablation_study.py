import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from gnn_model import ArchitectureGNN
import torch

def detect_arch_family(model_name):
    """Detect architecture family as a categorical feature for XGBoost."""
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
    if 'cnn' in model_name:
        return 'cnn_custom'
    return 'other'

def run_ablation_study():
    # 1. Load the merged dataset
    csv_files = glob.glob('data/raw/*.csv')
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    df = df.dropna(subset=['execution_time_ms'])
    df = df[df['execution_time_ms'] > 0]
    
    print(f"Dataset: {len(df)} rows, {df['model_name'].nunique()} models")
    print(f"Batch sizes: {sorted(df['batch_size'].unique())}")
    
    # 2. Feature engineering
    # Base features
    base_features = ['batch_size', 'total_parameters', 'model_size_mb']
    
    # Compute features
    if 'flops' not in df.columns:
        df['flops'] = df['total_parameters'] * 2
    if 'compute_memory_ratio' not in df.columns:
        df['compute_memory_ratio'] = df['flops'] / (df['model_size_mb'] * 1024**2 + 1)
    compute_features = base_features + ['flops', 'compute_memory_ratio']
    
    # Rich architecture features
    if 'param_ratio' not in df.columns:
        df['param_ratio'] = df['trainable_parameters'] / (df['total_parameters'] + 1)
    
    # Architecture family (categorical encoded)
    df['arch_family'] = df['model_name'].apply(detect_arch_family)
    df['arch_family_code'] = pd.Categorical(df['arch_family']).codes
    
    rich_features = compute_features + ['trainable_parameters', 'param_ratio', 'arch_family_code']
    
    # 3. BATCH-SIZE SPLIT
    train_batch_sizes = [1, 2, 4, 8]
    test_batch_sizes = [16, 32, 64, 128, 256]
    
    train_df = df[df['batch_size'].isin(train_batch_sizes)].copy()
    test_df = df[df['batch_size'].isin(test_batch_sizes)].copy()
    
    if len(test_df) == 0:
        # Fallback: try smaller test set
        test_batch_sizes = [16, 32, 64]
        test_df = df[df['batch_size'].isin(test_batch_sizes)].copy()
    
    print(f"\nSplit: Train on bs={train_batch_sizes} ({len(train_df)} rows)")
    print(f"       Test on bs={test_batch_sizes} ({len(test_df)} rows)")
    
    y_train = np.log1p(train_df['execution_time_ms'])
    y_test_true = test_df['execution_time_ms']
    
    if len(test_df) == 0:
        print("ERROR: No test data! Check batch sizes in dataset.")
        return
    
    results = []
    
    # Evaluate Conditions 1, 2, 3 using XGBoost
    for name, features in [('1. Base Params', base_features), 
                           ('2. Base + Compute', compute_features), 
                           ('3. Rich Arch + Family', rich_features)]:
        
        xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        xgb.fit(train_df[features], y_train)
        
        preds_log = xgb.predict(test_df[features])
        preds_log = np.clip(preds_log, -2, 10)
        preds = np.expm1(preds_log)
        
        mape = mean_absolute_percentage_error(y_test_true, preds) * 100
        results.append({'Condition': name, 'MAPE (%)': f"{mape:.2f}"})
        print(f"  {name}: MAPE = {mape:.2f}%")
        
    # Condition 4: Pure GNN
    print("\nEvaluating GNN conditions...")
    gnn = ArchitectureGNN()
    gnn_loaded = False
    try:
        gnn.load_state_dict(torch.load('models/gnn_predictor.pth', map_location='cpu', weights_only=True))
        gnn.eval()
        gnn_loaded = True
    except Exception as e:
        print(f"Could not load GNN: {e}")
    
    if gnn_loaded:
        from gnn_extractor import model_to_graph
        import torchvision.models as models
        
        model_builders = {
            'resnet18': lambda: models.resnet18(weights=None),
            'resnet50': lambda: models.resnet50(weights=None),
            'mobilenet_v2': lambda: models.mobilenet_v2(weights=None),
            'densenet121': lambda: models.densenet121(weights=None),
            'vgg16': lambda: models.vgg16(weights=None),
            'efficientnet_b0': lambda: models.efficientnet_b0(weights=None),
            'regnet_y_400mf': lambda: models.regnet_y_400mf(weights=None),
            'shufflenet_v2_x1_0': lambda: models.shufflenet_v2_x1_0(weights=None),
            'squeezenet1_0': lambda: models.squeezenet1_0(weights=None),
            'wide_resnet50_2': lambda: models.wide_resnet50_2(weights=None),
            'convnext_tiny': lambda: models.convnext_tiny(weights=None),
            'efficientnet_v2_s': lambda: models.efficientnet_v2_s(weights=None),
            'convnext_small': lambda: models.convnext_small(weights=None),
            'convnext_base': lambda: models.convnext_base(weights=None),
            'regnet_x_400mf': lambda: models.regnet_x_400mf(weights=None),
            'regnet_y_800mf': lambda: models.regnet_y_800mf(weights=None),
            'mnasnet1_0': lambda: models.mnasnet1_0(weights=None),
            'googlenet': lambda: models.googlenet(weights=None),
            'inception_v3': lambda: models.inception_v3(weights=None),
            'resnext50_32x4d': lambda: models.resnext50_32x4d(weights=None),
            'mobilenet_v3_large': lambda: models.mobilenet_v3_large(weights=None),
            'mobilenet_v3_small': lambda: models.mobilenet_v3_small(weights=None),
            'densenet169': lambda: models.densenet169(weights=None),
            'densenet201': lambda: models.densenet201(weights=None),
            'vgg19': lambda: models.vgg19(weights=None),
        }
        
        graph_cache = {}
        
        # Condition 4: Pure GNN
        gnn_errors = []
        for idx, row in test_df.iterrows():
            m_name = row['model_name']
            if m_name not in model_builders:
                continue
            if m_name not in graph_cache:
                net = model_builders[m_name]()
                graph_cache[m_name] = model_to_graph(net)
            
            graph = graph_cache[m_name]
            bs_tensor = torch.tensor([[float(row['batch_size'])]], dtype=torch.float32)
            
            with torch.no_grad():
                out = gnn(graph, bs_tensor)
            
            pred_ms = max(0.1, float(np.expm1(out[0, 0].item())))
            actual_ms = row['execution_time_ms']
            error = abs(pred_ms - actual_ms) / actual_ms
            gnn_errors.append(error)
            
        if gnn_errors:
            avg_mape = np.mean(gnn_errors) * 100
            results.append({'Condition': '4. Pure GNN', 'MAPE (%)': f"{avg_mape:.2f}"})
            print(f"  4. Pure GNN: MAPE = {avg_mape:.2f}%")
        else:
            results.append({'Condition': '4. Pure GNN', 'MAPE (%)': "N/A"})
        
        # Condition 5: GNN + Rich Features Combined (THE ACTUAL SYSTEM)
        # Use GNN embedding as an additional feature for XGBoost
        print("  Building GNN embeddings for Condition 5...")
        
        train_gnn_feats = []
        test_gnn_feats = []
        
        for split_df, feat_list in [(train_df, train_gnn_feats), (test_df, test_gnn_feats)]:
            for idx, row in split_df.iterrows():
                m_name = row['model_name']
                if m_name not in model_builders:
                    feat_list.append(0.0)
                    continue
                if m_name not in graph_cache:
                    net = model_builders[m_name]()
                    graph_cache[m_name] = model_to_graph(net)
                
                graph = graph_cache[m_name]
                bs_tensor = torch.tensor([[float(row['batch_size'])]], dtype=torch.float32)
                
                with torch.no_grad():
                    out = gnn(graph, bs_tensor)
                feat_list.append(float(out[0, 0].item()))  # log-space GNN prediction
        
        train_df_c5 = train_df[rich_features].copy()
        train_df_c5['gnn_embedding'] = train_gnn_feats
        
        test_df_c5 = test_df[rich_features].copy()
        test_df_c5['gnn_embedding'] = test_gnn_feats
        
        c5_features = rich_features + ['gnn_embedding']
        
        xgb_c5 = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        xgb_c5.fit(train_df_c5[c5_features], y_train)
        
        preds_log = xgb_c5.predict(test_df_c5[c5_features])
        preds_log = np.clip(preds_log, -2, 10)
        preds = np.expm1(preds_log)
        
        mape = mean_absolute_percentage_error(y_test_true, preds) * 100
        results.append({'Condition': '5. GNN + Rich (Blink)', 'MAPE (%)': f"{mape:.2f}"})
        print(f"  5. GNN + Rich (Blink): MAPE = {mape:.2f}%")
    else:
        results.append({'Condition': '4. Pure GNN', 'MAPE (%)': "Error"})
        results.append({'Condition': '5. GNN + Rich (Blink)', 'MAPE (%)': "Error"})
    
    # --- MEMORY PREDICTION MAPE ---
    print("\n--- Memory Prediction Accuracy ---")
    if 'peak_memory_mb' in df.columns:
        mem_df = df.dropna(subset=['peak_memory_mb'])
        mem_df = mem_df[mem_df['peak_memory_mb'] > 0]
        mem_train = mem_df[mem_df['batch_size'].isin(train_batch_sizes)]
        mem_test = mem_df[mem_df['batch_size'].isin(test_batch_sizes)]
        
        if len(mem_test) > 0:
            xgb_mem = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
            xgb_mem.fit(mem_train[rich_features], np.log1p(mem_train['peak_memory_mb']))
            mem_preds = np.expm1(np.clip(xgb_mem.predict(mem_test[rich_features]), -2, 12))
            mem_mape = mean_absolute_percentage_error(mem_test['peak_memory_mb'], mem_preds) * 100
            print(f"  Memory MAPE (Rich Arch): {mem_mape:.2f}%")
            results.append({'Condition': 'Memory (Rich Arch)', 'MAPE (%)': f"{mem_mape:.2f}"})
    
    # --- CI CALIBRATION ---
    print("\n--- CI Calibration Check ---")
    xgb_final = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_final.fit(train_df[rich_features], y_train)
    preds_log = np.clip(xgb_final.predict(test_df[rich_features]), -2, 10)
    preds_final = np.expm1(preds_log)
    
    # Simple CI: ±15% around prediction
    lower = preds_final * 0.85
    upper = preds_final * 1.15
    y_actual = test_df['execution_time_ms'].values
    coverage = np.mean((y_actual >= lower) & (y_actual <= upper))
    print(f"  ±15% CI coverage: {coverage:.1%} (target ~80%)")
    
    # Try ±30%
    lower30 = preds_final * 0.70
    upper30 = preds_final * 1.30  
    coverage30 = np.mean((y_actual >= lower30) & (y_actual <= upper30))
    print(f"  ±30% CI coverage: {coverage30:.1%}")
    
    # Output Table II
    res_df = pd.DataFrame(results)
    print("\n--- TABLE II: Feature Ablation Study (Batch-Size Split) ---")
    print(res_df.to_string(index=False))
    res_df.to_csv('results/ablation_study_table.csv', index=False)
    print("\nSaved to results/ablation_study_table.csv")

if __name__ == '__main__':
    run_ablation_study()
