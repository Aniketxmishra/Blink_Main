import joblib
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from train_eval_blink import FEATURE_COLS, TARGET_COL, feature_engineering, load_data

print("Loading and applying identical feature engineering pipeline...")
df = load_data()
feature_df = feature_engineering(df)

# Sanity check: verify no transformers snuck in
LLM_MODELS = ['bert', 'gpt2', 'roberta', 'vit', 'swin', 'maxvit']
mask = feature_df['model_name'].str.lower().str.contains('|'.join(LLM_MODELS), case=False, na=False)
if mask.any():
    print(f"Warning: Found {mask.sum()} transformer rows, dropping them...")
    feature_df = feature_df[~mask].copy()

# Ensure we use the exact same feature columns actually generated and present
available_cols = [c for c in FEATURE_COLS if c in feature_df.columns]
X = feature_df[available_cols]
y = np.log1p(feature_df[TARGET_COL])  # split the identical way the trainer did

# Same 80/20 random split with same seed
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Recover the original test values (un-log) for true MAPE calculation
_, test_df = train_test_split(feature_df, test_size=0.2, random_state=42)
y_test = test_df[TARGET_COL].values

print("\nLoading XGBoost model...")
try:
    model = joblib.load('models/xgb_latency.pkl')
except FileNotFoundError:
    print("Error: models/xgb_latency.pkl not found! Did train_eval_blink.py finish saving it?")
    exit(1)

# Predict median (quantile 0.5 head or objective='reg:quantileerror')
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_pred = np.maximum(1.0, y_pred) # enforce positive bound

# Calculate True MAPE
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"\n[SUCCESS] Independent MAPE: {mape:.2f}%")
print(f"   Test samples:     {len(y_test)}")
print(f"   Mean true latency: {y_test.mean():.2f}ms")
print(f"   Mean pred latency: {y_pred.mean():.2f}ms")

# Breakdown by architecture family
df_test = X_test.copy()
df_test['true'] = y_test
df_test['pred'] = y_pred
df_test['model_name'] = test_df['model_name'].values
df_test['ape'] = abs(df_test['true'] - df_test['pred']) / df_test['true'] * 100

print("\n=== MAPE by Architecture Family ===")
families = {
    'ResNet':       'resnet',
    'EfficientNet': 'efficientnet',
    'VGG':          'vgg',
    'MobileNet':    'mobilenet',
    'ConvNeXt':     'convnext',
    'ResNeXt':      'resnext',
    'SqueezeNet':   'squeezenet',
}
for name, key in families.items():
    mask = df_test['model_name'].str.contains(key, case=False)
    if mask.sum() > 0:
        print(f"  {name:12s}: {df_test.loc[mask,'ape'].mean():.2f}%  ({mask.sum()} samples)")
