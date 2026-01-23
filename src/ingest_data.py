import os
import pandas as pd
import numpy as np

def ingest():
    MLOPS_ROOT = r"D:\MLOps"
    RAW_DATA_DIR = os.path.join(MLOPS_ROOT, 'input_data', 'raw')
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    target_path = os.path.join(RAW_DATA_DIR, 'raw_data.csv')
    
    # Check if real data exists, otherwise create a VALID sample
    if not os.path.exists(target_path):
        print("ALERT: No raw file found. Creating a Lesson 15 compatible sample...")
        data = {
            'Date': pd.date_range(start='2025-01-01', periods=20, freq='D'),
            'Sales': np.random.randint(100, 500, size=20),
            'Temperature': np.random.randint(15, 35, size=20),
            'Category': np.random.choice(['Tech', 'Office', 'Furniture'], 20),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 20),
            'Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], 20)
        }
        df = pd.DataFrame(data)
        df.to_csv(target_path, index=False)
        print(f"SUCCESS: Sample data created with all required columns at {target_path}")
    else:
        print("SUCCESS: Real raw data detected in the official warehouse.")

if __name__ == "__main__":
    ingest()