import pandas as pd
import numpy as np
import os
import json
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def load_data(data_dir='data/enriched'):
    """Load feature data from CSV and JSON files"""
    if not os.path.exists(data_dir):
        # Fallback to data/processed if data/enriched doesn't exist
        print(f"Directory {data_dir} does not exist. Falling back to data/processed")
        data_dir = 'data/processed'
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist.")
            return None

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    if not csv_files and not json_files:
        print(f"No CSV or JSON files found in {data_dir}.")
        return None

    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            all_data.extend(df.to_dict('records'))
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    df = pd.DataFrame(all_data)
    
    # Filter noisy timings (but preserve rows that don't have timing_cv, like LLMs)
    if 'timing_cv' in df.columns:
        initial_len = len(df)
        df = df[df['timing_cv'].fillna(0) <= 0.15]
        print(f"Dropped {initial_len - len(df)} rows due to timing_cv > 0.15")
        
    print(f"DataFrame columns: {df.columns.tolist()}")
    return df

def prepare_features(df):
    """Prepare features for model training"""
    # Group by model_name and batch_size to get unique model configurations
    # For each configuration, we'll use the model parameters as features
    # and execution time as the target
    
    features = []
    for (model_name, batch_size), group in df.groupby(['model_name', 'batch_size']):
        # Get the first row for this configuration
        row = group.iloc[0]
        
        # Create feature dictionary
        feature_dict = {
            'model_name': model_name,
            'batch_size': batch_size,
            'total_parameters': row['total_parameters'],
            'trainable_parameters': row['trainable_parameters'],
            'model_size_mb': row['model_size_mb'],
            'execution_time_ms': row['execution_time_ms']
        }
        
        # Add enriched features
        enriched_cols = ['flops', 'compute_memory_ratio', 'memory_read_write_ratio',
                         'num_conv_layers', 'num_fc_layers', 'num_bn_layers',
                         'avg_conv_kernel_size', 'max_conv_channels', 'total_conv_params',
                         'max_fc_size', 'total_fc_params', 'model_depth',
                         'tflops_fp32', 'memory_bandwidth_gbps', 'sm_count',
                         # LLM-specific features (will be 0 for CNN rows — that's fine)
                         'vocab_size', 'seq_len', 'kv_cache_size_mb', 'quantization_bits']
        
        for col in enriched_cols:
            feature_dict[col] = row.get(col, 0)

        # LLM-specific targets (0 for CNN rows)
        feature_dict['prefill_time_ms'] = row.get('prefill_time_ms', 0)
        feature_dict['decode_time_ms']  = row.get('decode_time_ms',  0)
        feature_dict['is_llm']          = int(bool(row.get('is_llm', False)))
        
        features.append(feature_dict)

    
    feature_df = pd.DataFrame(features)
    
    return feature_df

def train_models(df, target='execution_time_ms', test_size=0.2, random_state=42):
    """Train and evaluate multiple regression models"""
    # Define features and target using rich feature set
    feature_cols = [
        'batch_size',
        'flops',
        'compute_memory_ratio',
        'num_conv_layers',
        'num_fc_layers',
        'num_bn_layers',
        'avg_conv_kernel_size',
        'max_conv_channels',
        'total_conv_params',
        'total_fc_params',
        'model_depth',
        'model_size_mb',
        # Hardware generalization features
        'tflops_fp32',
        'memory_bandwidth_gbps',
        'sm_count',
        # LLM-specific features
        'vocab_size',
        'seq_len',
        'kv_cache_size_mb',
        'quantization_bits',
    ]

    # Filter only available columns to prevent errors
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    
    # Log-transform target
    y = np.log1p(df[target])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Optuna tuning for XGBoost
    print("Tuning XGBoost hyperparameters with Optuna...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': random_state,
        }
        # Use simple validation split for tuning
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        # We optimize for RMSE on the log scale
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=30)
    best_xgb_params = study.best_params
    best_xgb_params['random_state'] = random_state
    print(f"Best XGBoost params: {best_xgb_params}")

    # Define models (only tree based models which can handle log-transforms natively without forcing exponential curves)
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
        'XGBoost (Tuned)': xgb.XGBRegressor(**best_xgb_params)
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    # Keep original y_test for metrics
    y_test_orig = np.expm1(y_test)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        
        # Calculate metrics on original scale
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        mae = mean_absolute_error(y_test_orig, y_pred)
        r2 = r2_score(y_test_orig, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        trained_models[name] = model
    
    # Find best model
    best_model_name = min(results, key=lambda k: results[k]['RMSE'])
    print(f"Best model: {best_model_name}")
    
    print("\nTraining Uncertainty Models (Unified Quantile Regression)...")
    
    # 1. Train the Median (target for point prediction)
    model_median = xgb.XGBRegressor(**best_xgb_params, objective='reg:quantileerror', quantile_alpha=0.5)
    model_median.fit(X_train, y_train)
    models['Median Quantile (0.5)'] = model_median
    
    # Override best model name so the app knows to use the quantile predictor
    best_model_name = 'Median Quantile (0.5)'
    
    # 2. Train lower bound (10th percentile)
    model_lower = xgb.XGBRegressor(**best_xgb_params, objective='reg:quantileerror', quantile_alpha=0.1)
    model_lower.fit(X_train, y_train)
    
    # 3. Train upper bound (90th percentile)
    model_upper = xgb.XGBRegressor(**best_xgb_params, objective='reg:quantileerror', quantile_alpha=0.9)
    model_upper.fit(X_train, y_train)
    
    bounds_models = {
        'lower': model_lower,
        'upper': model_upper
    }
    
    return models, bounds_models, results, X_test, y_test_orig, best_model_name

def evaluate_ood(models, bounds_models, feature_cols, df):
    """Run strict OOD tests based on unseen batch sizes and model families."""
    from model_analyser import ModelAnalyzer
    
    # Needs re-loading with full df
    df_clean = df.copy()
    
    print("\n--- OOD Evaluation (Unseen Batch Sizes: Train <= 8, Test >= 16) ---")
    train_df, test_df = ModelAnalyzer.ood_extrapolation_split(df_clean, mode='batch_size', train_thresh=8)
    if len(test_df) > 0:
        X_test_ood = test_df[feature_cols]
        y_test_ood = test_df['execution_time_ms']
        
        # Predict 
        median   = np.expm1(models['Median Quantile (0.5)'].predict(X_test_ood))
        q_lower  = np.expm1(bounds_models['lower'].predict(X_test_ood))
        q_upper  = np.expm1(bounds_models['upper'].predict(X_test_ood))
        
        # Enforce guarantees
        lower = np.maximum(1.0, q_lower)
        mid = np.maximum(lower, median)
        upper = np.maximum(mid, q_upper)
        
        mape = np.mean(np.abs((y_test_ood - mid) / y_test_ood)) * 100
        coverage = np.mean((y_test_ood >= lower) & (y_test_ood <= upper)) * 100
        pinball = ModelAnalyzer.interval_pinball_loss(y_test_ood, lower, upper)
        
        print(f"OOD BS>8 -> MAPE: {mape:.2f}%, Coverage: {coverage:.1f}%, Interval Loss: {pinball:.2f}")

def visualize_results(models, results, X_test, y_test, best_model_name, bounds_models=None):
    """Create visualizations of model performance and feature importance"""
    # Plot model comparison
    plt.figure(figsize=(12, 6))
    
    metrics = ['RMSE', 'MAE', 'R2']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        values = [results[model][metric] for model in results if metric in results[model]]
        models_keys = [model for model in results if metric in results[model]]
        if len(values) > 0:
            sns.barplot(x=models_keys, y=values)
            plt.title(f'Model Comparison - {metric}')
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Plot predicted vs actual for best point model
    best_model = models.get(best_model_name, list(models.values())[0])
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    # Add bounds evaluation if they exist
    if bounds_models:
        from model_analyser import ModelAnalyzer
        q_lower = np.expm1(bounds_models['lower'].predict(X_test))
        q_upper = np.expm1(bounds_models['upper'].predict(X_test))
        
        lower = np.maximum(1.0, q_lower)
        mid = np.maximum(lower, y_pred)
        upper = np.maximum(mid, q_upper)
        
        coverage = np.mean((y_test >= lower) & (y_test <= upper)) * 100
        pinball = ModelAnalyzer.interval_pinball_loss(y_test, lower, upper)
        print(f"\nGeneral Validation -> 80% CI Coverage: {coverage:.1f}%, Interval Loss: {pinball:.2f}")
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred)
    # optionally plot error bars if bounds exist
    if bounds_models:
        plt.errorbar(x=y_test, y=mid, yerr=[mid-lower, upper-mid], fmt='none', ecolor='orange', alpha=0.3, zorder=0)
        
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.title(f'Predicted vs Actual - {best_model_name}')
    plt.savefig('predicted_vs_actual.png')
    
    # Plot feature importance for tree-based models
    if best_model_name in ['Random Forest', 'XGBoost (Tuned)']:
        # For pipelines, the model is the last step
        if hasattr(best_model, 'steps'):
            model = best_model.steps[-1][1]
        else:
            model = best_model
            
        feature_cols = [c for c in X_test.columns]
        
        plt.figure(figsize=(10, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')

def save_best_model(models, bounds_models, best_model_name, output_dir='models'):
    """Save the best model and uncertainty bounds for later use"""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"{best_model_name.lower().replace(' ', '_')}_model.joblib")
    joblib.dump(models[best_model_name], model_path)
    
    # Save bounds models
    joblib.dump(bounds_models['lower'], os.path.join(output_dir, 'execution_lower_model.joblib'))
    joblib.dump(bounds_models['upper'], os.path.join(output_dir, 'execution_upper_model.joblib'))
    
    print(f"Best model saved to {model_path}")
    print(f"Confidence bound models saved to {output_dir}")
    
    return model_path

def main():
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Prepare features
    feature_df = prepare_features(df)
    
    # Train models
    models, bounds_models, results, X_test, y_test, best_model_name = train_models(feature_df)
    
    # Define features and target using rich feature set
    feature_cols = [
        'batch_size', 'flops', 'compute_memory_ratio', 'num_conv_layers',
        'num_fc_layers', 'num_bn_layers', 'avg_conv_kernel_size', 'max_conv_channels',
        'total_conv_params', 'total_fc_params', 'model_depth', 'model_size_mb',
        'tflops_fp32', 'memory_bandwidth_gbps', 'sm_count',
        # LLM features
        'vocab_size', 'seq_len', 'kv_cache_size_mb', 'quantization_bits',
    ]

    feature_cols = [c for c in feature_cols if c in feature_df.columns]
    
    # Run OOD Extrapolation Checks
    evaluate_ood(models, bounds_models, feature_cols, feature_df)
    
    # Visualize results
    visualize_results(models, results, X_test, y_test, best_model_name, bounds_models)
    
    # Save best model
    model_path = save_best_model(models, bounds_models, best_model_name)
    
    print("Model training and evaluation complete!")
    print(f"Results saved to results/ directory")
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    main()
