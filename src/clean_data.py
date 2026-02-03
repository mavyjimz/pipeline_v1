import pandas as pd
import os

output_file = "input_data/processed/sales_summary.csv"
raw_path = "input_data/raw/superstore_sales.csv"

def clean_and_prepare():
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    # Load with specific encoding for superstore data
    df = pd.read_csv(raw_path, encoding='latin1')

    # Convert Sales to numeric and remove rows with errors
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df = df.dropna(subset=['Sales'])

    # Select ONLY numeric columns for the model
    # We drop text columns like 'Order ID', 'Customer Name', etc.
    numeric_df = df.select_dtypes(include=['number'])

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save only the numeric data for training
    numeric_df.to_csv(output_file, index=False)
    print(f"Success: {output_file} updated with numeric features only.")

if __name__ == "__main__":
    clean_and_prepare()