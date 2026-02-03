import pandas as pd
import os

# Standardized Relative Paths
output_file = "input_data/processed/sales_summary.csv"
raw_path = "input_data/raw/superstore_sales.csv"

def clean_data():
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    print(f"Loading data from: {raw_path}")
    # Logic for handling superstore_sales.csv errors goes here
    df = pd.read_csv(raw_path, encoding='latin1')
    
    # Save processed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Success: {output_file} has been updated.")

if __name__ == "__main__":
    clean_data()