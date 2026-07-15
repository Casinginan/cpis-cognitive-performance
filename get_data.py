import pandas as pd
import os

real_path = "data/sleep_data_real.csv"

if os.path.exists(real_path):
    df = pd.read_csv(real_path)
    print(f"Real dataset found: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
else:
    print("Real dataset not found - run this after uploading the CSV")

