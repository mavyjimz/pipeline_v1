import pandas as pd
import numpy as np
import os

def clean_data():
    raw_path = r"D:\MLOps\input_data\raw\superstore_sales.csv"
    processed_path = r"D:\MLOps\input_data\processed\sales_summary.csv"
    
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    df = pd.read_csv(raw_path)

    # 1. THE SCOUTED TARGETS (Match capitalization exactly!)
    # We found these in image_5efc19.png
    category_cols = ['Segment', 'Region', 'Category']
    numeric_cols = ['Sales', 'Temperature']

    print(f"--- [FACTORY START]: Processing {len(df)} rows ---")

    # 2. THE TRANSFORMER (One-Hot Encoding)
    # This line forces pandas to treat them as categories
    df_encoded = pd.get_dummies(df, columns=category_cols, dummy_na=False)

    # DEBUG: Let's see if the expansion actually happened
    print(f"DEBUG: Total columns after expansion: {len(df_encoded.columns)}")

    # 3. THE NUMERIC SHIELD
    # Keep only Sales, Temp, and our brand new Encoded columns
    df_final = df_encoded.select_dtypes(include=[np.number]).dropna()

    # 4. EXPORT
    df_final.to_csv(processed_path, index=False)
    
    print(f"SUCCESS: Factory output secured at {processed_path}")
    # (-1 to exclude the 'Sales' target from the feature count)
    print(f"New Feature Count: {len(df_final.columns) - 1}")
    print(f"Detected Columns: {df_final.columns.tolist()[:10]}... (Total: {len(df_final.columns)})")

if __name__ == "__main__":
    clean_data()