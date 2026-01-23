import os
import pandas as pd

# 1. DEFINE ROOTS (The Standard Law)
# This assumes the script is in D:/MLOps/projects/pipeline_v1/src
# We want to go up to D:/MLOps
MLOPS_ROOT = r"D:\MLOps"

# 2. ALIGN WITH OFFICIAL HIERARCHY
# We removed the 'data' middle-folder to stop the "Ghost" rebellion
RAW_DATA_DIR = os.path.join(MLOPS_ROOT, 'input_data', 'raw')

def ingest():
    # Ensure the standard directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Path for our source file
    target_path = os.path.join(RAW_DATA_DIR, 'raw_data.csv')
    
    print(f"INGESTION: Moving raw data to {target_path}...")
    
    # --- YOUR INGESTION LOGIC HERE ---
    # For now, let's assume we are just checking if it's there or creating a placeholder
    if not os.path.exists(target_path):
        print("ALERT: No raw file found. Please place your raw CSV in 'input_data/raw/'")
    else:
        print("SUCCESS: Raw data is in the official warehouse.")

if __name__ == "__main__":
    ingest()