import glob

import pandas as pd

df = None
files = glob.glob('data/enriched/*.csv')
if not files:
    print("No enriched files found, trying raw")
    files = glob.glob('data/raw/*.csv')

if files:
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        print(f"Rows: {len(df)}")
        print(f"Archs: {df['model_name'].nunique()}")
        if 'device' in df.columns:
            print(f"GPUs: {df['device'].nunique()}")
        elif 'gpu_name' in df.columns:
            print(f"GPUs: {df['gpu_name'].nunique()}")
        else:
            print("GPUs: 0")
        print("\nSchema of first file (" + files[0] + "):")
        print(dfs[0].dtypes)
else:
    print("No CSV files found")
