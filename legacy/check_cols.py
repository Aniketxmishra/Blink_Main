import pandas as pd
import glob
import json

raw_files = glob.glob('data/raw/*.csv')
if raw_files:
    df = pd.read_csv(raw_files[0])
    print("Raw CSV columns:", list(df.columns))
    print("Raw CSV len:", len(df))

processed_files = glob.glob('data/processed/*.csv')
if processed_files:
    df = pd.read_csv(processed_files[0])
    print("Processed CSV columns:", list(df.columns))
    print("Processed CSV len:", len(df))
