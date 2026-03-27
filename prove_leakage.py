import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb


def load_data(data_dir='data/enriched'):
    if not os.path.exists(data_dir): 
        data_dir = 'data/processed'
    all_data = []
    for f in os.listdir(data_dir):
        if f.endswith('.csv'):
            all_data.extend(pd.read_csv(os.path.join(data_dir, f)).to_dict('records'))
        elif f.endswith('.json'):
            with open(os.path.join(data_dir, f)) as fp:
                all_data.extend(json.load(fp))
    df = pd.DataFrame(all_data)
    if 'timing_cv' in df.columns: 
        df = df[df['timing_cv'] <= 0.15]
    return df

if __name__ == '__main__':
    df = load_data()
    feature_cols = ['batch_size', 'flops', 'num_conv_layers', 'model_size_mb', 'compute_memory_ratio']
    
    train_df_strict = df[df['batch_size'] <= 8]
    test_df_ood = df[df['batch_size'] >= 16]
    X_train_strict = train_df_strict[feature_cols]
    y_test_ood = test_df_ood['execution_time_ms']
    X_ood = test_df_ood[feature_cols]
    
    # 1. Target as log(latency/batch_size)
    y_train_per_sample = np.log1p(train_df_strict['execution_time_ms'] / train_df_strict['batch_size'])
    model_per_sample = xgb.XGBRegressor()
    model_per_sample.fit(X_train_strict, y_train_per_sample)
    preds_per_sample = np.expm1(model_per_sample.predict(X_ood)) * X_ood['batch_size']
    mape_per_sample = np.mean(np.abs((y_test_ood - preds_per_sample) / y_test_ood)) * 100
    print(f"STRICT OOD MAPE (Target = latency/BS): {mape_per_sample:.2f}%")
    
    # 2. Linear Booster
    y_train_strict = np.log1p(train_df_strict['execution_time_ms'])
    model_linear = xgb.XGBRegressor(booster='gblinear')
    model_linear.fit(X_train_strict, y_train_strict)
    preds_linear = np.expm1(model_linear.predict(X_ood))
    mape_linear = np.mean(np.abs((y_test_ood - preds_linear) / y_test_ood)) * 100
    print(f"STRICT OOD MAPE (Linear Booster): {mape_linear:.2f}%")
