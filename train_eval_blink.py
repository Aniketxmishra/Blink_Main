import pandas as pd
import numpy as np
import os
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna

warnings.filterwarnings('ignore', category=UserWarning)

# --- HYPERPARAMETERS & CONFIG (Agent can modify these!) ---
RANDOM_STATE = 42
TARGET_COL = 'execution_time_ms'
FEATURE_COLS = [
    'batch_size', 'flops', 'compute_memory_ratio', 'num_conv_layers',
    'num_fc_layers', 'num_bn_layers', 'avg_conv_kernel_size', 'max_conv_channels',
    'total_conv_params', 'total_fc_params', 'model_depth', 'model_size_mb',
    'tflops_fp32', 'memory_bandwidth_gbps', 'sm_count',
    'flops_per_mb', 'params_per_layer', 'conv_to_fc_ratio', 'flops_to_bandwidth',
    'compute_intensity_score', 'depth_complexity_penalty'
]
# For faster agent iterations, we might reduce optuna trials
N_OPTUNA_TRIALS = 10 

def load_data(data_dir='data/enriched'):
    if not os.path.exists(data_dir):
        data_dir = 'data/processed'
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Neither data/enriched nor data/processed found.")

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(data_dir, csv_file))
        all_data.extend(df.to_dict('records'))

    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    for json_file in json_files:
        with open(os.path.join(data_dir, json_file), 'r') as f:
            all_data.extend(json.load(f))

    df = pd.DataFrame(all_data)
    if 'timing_cv' in df.columns:
        df = df[df['timing_cv'] <= 0.15]
    return df

def feature_engineering(df):
    """
    Agent can modify this function to invent new features,
    combine existing ones, or apply non-linear transformations!
    """
    features = []
    for (model_name, batch_size), group in df.groupby(['model_name', 'batch_size']):
        row = group.iloc[0]
        feature_dict = {
            'model_name': model_name,
            'batch_size': batch_size,
            'total_parameters': row.get('total_parameters', 0),
            'execution_time_ms': row.get('execution_time_ms', 0)
        }
        for col in FEATURE_COLS:
            if col != 'batch_size':
                feature_dict[col] = row.get(col, 0)
        
        # AGENT: Add custom feature engineering here!
        # Ratio of compute to size 
        feature_dict['flops_per_mb'] = row.get('flops', 0) / max(0.001, row.get('model_size_mb', 1))
        # Total parameters per depth layer
        feature_dict['params_per_layer'] = row.get('total_parameters', 0) / max(1, row.get('model_depth', 1))
        # Convolution intensity vs FC intensity
        feature_dict['conv_to_fc_ratio'] = row.get('total_conv_params', 0) / max(1, row.get('total_fc_params', 1))
        # Hardware bandwidth utilization potential
        feature_dict['flops_to_bandwidth'] = row.get('flops', 0) / max(0.001, row.get('memory_bandwidth_gbps', 1))
        
        # EXPERIMENT 7: Non-linear composite features
        feature_dict['compute_intensity_score'] = feature_dict['flops_per_mb'] * feature_dict['conv_to_fc_ratio']
        feature_dict['depth_complexity_penalty'] = feature_dict['params_per_layer'] * np.log1p(row.get('model_depth', 1))
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)

def evaluate_ood_split(df, feature_cols):
    """
    Evaluate the model on Out-Of-Distribution (OOD) data.
    We train on batch sizes <= 8 and test on batch sizes >= 16.
    """
    train_df = df[df['batch_size'] <= 8].copy()
    test_df = df[df['batch_size'] >= 16].copy()
    
    if len(test_df) == 0 or len(train_df) == 0:
        raise ValueError("Not enough data to perform OOD split on batch sizes.")

    X_train = train_df[feature_cols]
    y_train = np.log1p(train_df[TARGET_COL])
    
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL] # Keep original scale for MAPE
    
    # Optuna tuning
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.4, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': RANDOM_STATE,
        }
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Increased trials to find better params
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS + 10) 
    
    best_params = study.best_params
    best_params['random_state'] = RANDOM_STATE
    
    # Train final model on full train_df
    final_model = xgb.XGBRegressor(**best_params, objective='reg:quantileerror', quantile_alpha=0.48)
    final_model.fit(X_train, y_train)
    
    # Predict on OOD test data
    preds_log = final_model.predict(X_test)
    preds = np.expm1(preds_log)
    
    # Enforce positive bounds
    preds = np.maximum(1.0, preds)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    
    return mape

def main():
    print("Loading data...")
    df = load_data()
    
    print("Applying feature engineering...")
    feature_df = feature_engineering(df)
    
    # Filter features that are actually present
    available_features = [c for c in FEATURE_COLS if c in feature_df.columns]
    
    print(f"Training on {len(df)} samples and evaluating OOD MAPE...")
    try:
        mape_score = evaluate_ood_split(feature_df, available_features)
        print(f"Final OOD MAPE: {mape_score:.2f}%")
        
        # This is the single metric the autoresearch agent will parse. LOWER is better.
        print(f"SCORE: {mape_score:.4f}") 
    except Exception as e:
        print(f"Error during evaluation: {e}")
        # Return a terrible score if something breaks
        print(f"SCORE: 9999.0")

if __name__ == "__main__":
    main()
