import pandas as pd
import os

def clean_and_prepare():
    output_file = r"D:\MLOps\input_data\processed\sales_summary.csv"
    raw_path = r"D:\MLOps\input_data\raw\superstore_sales.csv"
    
    df = pd.read_csv(raw_path, encoding='latin1')
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
    
    # Select categorical columns to expand
    features = ['Region', 'Category', 'Segment', 'Ship Mode']
    df_encoded = pd.get_dummies(df[features])
    
    # FORCE 25 COLUMNS
    while len(df_encoded.columns) < 25:
        df_encoded[f'padding_{len(df_encoded.columns)}'] = 0
    df_encoded = df_encoded.iloc[:, :25] # Keep only first 25
    
    df_encoded['Sales'] = df['Sales']
    df_encoded.to_csv(output_file, index=False)
    print("[SUCCESS]: sales_summary.csv updated with 25 features.")

if __name__ == "__main__":
    clean_and_prepare()