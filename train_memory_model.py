import os
import pandas as pd
import numpy as np
import joblib
import optuna
import xgboost as xgb
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)

def load_data(data_path='data/enriched/enriched_data.csv'):
    """Load enriched feature data."""
    if not os.path.exists(data_path):
        print(f"File {data_path} does not exist.")
        return None
    df = pd.read_csv(data_path)
    if 'peak_memory_mb' not in df.columns:
        print("Error: 'peak_memory_mb' not found in the dataset. Did you run collect_data.py with GPU enabled?")
        return None
    # Filter out entries where peak memory is 0 or NaN
    df = df[df['peak_memory_mb'] > 0]
    
    # Filter noisy timings
    if 'timing_cv' in df.columns:
        initial_len = len(df)
        df = df[df['timing_cv'] <= 0.15]
        print(f"Dropped {initial_len - len(df)} rows due to timing_cv > 0.15")
        
    return df

def train_memory_model(df, target='peak_memory_mb', test_size=0.2, random_state=42):
    """Train XGBoost model for memory prediction."""
    # Define features based on Priority 1 rich feature set
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
        'model_size_mb'
    ]
    
    # Filter only available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("Tuning XGBoost hyperparameters with Optuna for Memory Prediction...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': random_state,
        }
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
        
    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    best_params['random_state'] = random_state
    
    print(f"Best XGBoost params for Memory: {best_params}")
    
    # Train final model on all training data
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"RMSE: {rmse:.2f} MB")
    print(f"MAE:  {mae:.2f} MB")
    print(f"R2:   {r2:.4f}")
    
    print("\nTraining Uncertainty Models (Quantile Regression)...")
    # Train lower bound (10th percentile)
    model_lower = xgb.XGBRegressor(**best_params, objective='reg:quantileerror', quantile_alpha=0.1)
    model_lower.fit(X_train, y_train)
    
    # Train upper bound (90th percentile)
    model_upper = xgb.XGBRegressor(**best_params, objective='reg:quantileerror', quantile_alpha=0.9)
    model_upper.fit(X_train, y_train)
    
    bounds_models = {
        'lower': model_lower,
        'upper': model_upper
    }
    
    return model, bounds_models, X_test, y_test, feature_cols

def visualize_memory_model(model, X_test, y_test, feature_cols):
    """Plot results for the memory model."""
    os.makedirs('results', exist_ok=True)
    
    # Actual vs Predicted
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.7)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Peak Memory (MB)')
    plt.ylabel('Predicted Peak Memory (MB)')
    plt.title('Memory Model: Predicted vs Actual')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/memory_predicted_vs_actual.png')
    
    # Feature Importance
    plt.figure(figsize=(10, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Memory Model: Feature Importance')
    plt.tight_layout()
    plt.savefig('results/memory_feature_importance.png')

def main():
    print("Loading enriched data...")
    df = load_data()
    
    if df is None or len(df) == 0:
        print("Failed to load suitable data. Exiting.")
        return
        
    print("Training dedicated Memory Prediction model...")
    model, bounds_models, X_test, y_test, feature_cols = train_memory_model(df)
    
    visualize_memory_model(model, X_test, y_test, feature_cols)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/memory_model.joblib'
    joblib.dump(model, model_path)
    joblib.dump(bounds_models['lower'], 'models/memory_lower_model.joblib')
    joblib.dump(bounds_models['upper'], 'models/memory_upper_model.joblib')
    print(f"\nMemory model and confidence bounds saved to models/")

if __name__ == "__main__":
    main()
