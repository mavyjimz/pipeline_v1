import pandas as pd
import numpy as np
import os

def clean_data():
    MLOPS_ROOT = r"D:\MLOps"
    input_file = os.path.join(MLOPS_ROOT, 'input_data', 'raw', 'raw_data.csv')
    output_file = os.path.join(MLOPS_ROOT, 'input_data', 'processed', 'sales_summary.csv')
    
    if not os.path.exists(input_file):
        print(f"ERROR: No raw data at {input_file}")
        return

    df = pd.read_csv(input_file)

    # 1. FORCE NUMERIC: Convert everything possible to a number
    # This catches 'Sales', 'Temperature', and any others that exist
    for col in df.columns:
        # If the column name sounds like math, force it to be math
        if col in ['Sales', 'Temperature', 'Quantity', 'Discount', 'Profit']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. DATE HANDLING
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values(by='Date')

    # 3. THE SAFETY FILTER: Drop rows that became NaNs during conversion
    df = df.dropna(subset=['Sales']) 

    df.to_csv(output_file, index=False)
    print(f"SUCCESS: Factory output secured at {output_file}")
    print(f"Detected Numeric Columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")

if __name__ == "__main__":
    clean_data()