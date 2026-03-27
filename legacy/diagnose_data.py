import glob

import pandas as pd

csv_files = glob.glob('data/raw/*.csv')
dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)
df = df.dropna(subset=['execution_time_ms'])
df = df[df['execution_time_ms'] > 0]

print(f"Total rows: {len(df)}")
print(f"Unique models: {df['model_name'].nunique()}")
print("\nRows per model:")
print(df['model_name'].value_counts().to_string())
print(f"\nBatch sizes: {sorted(df['batch_size'].unique())}")
print(f"\nExec time range: {df['execution_time_ms'].min():.2f} - {df['execution_time_ms'].max():.2f} ms")
print(f"\nColumn names: {list(df.columns)}")
