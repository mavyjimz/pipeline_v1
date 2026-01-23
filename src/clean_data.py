import pandas as pd
import numpy as np
import os

def clean_data():
    # 1. DEFINE ROOTS
    MLOPS_ROOT = r"D:\MLOps"
    
    # 2. ALIGN WITH OFFICIAL WAREHOUSE
    input_file = os.path.join(MLOPS_ROOT, 'input_data', 'raw', 'raw_data.csv')
    output_dir = os.path.join(MLOPS_ROOT, 'input_data', 'processed')
    output_file = os.path.join(output_dir, 'sales_summary.csv')
    
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        
        # --- NEW: NUMERIC FORCE (UNLOCK LESSON 18) ---
        # List the columns we NEED for the model to see
        target_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Temperature']
        
        for col in target_cols:
            if col in df.columns:
                # errors='coerce' turns text-errors into NaNs, 
                # which our train_model.py 'Janitor' will then sweep away.
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Perform date cleaning as before
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        
        # Save to the OFFICIAL "Certified Clean" location
        df.to_csv(output_file, index=False)
        print(f"SUCCESS: Data cleaned and forced to numeric. Saved to {output_file}")
    else:
        print(f"ERROR: Could not find {input_file}")

if __name__ == "__main__":
    clean_data()