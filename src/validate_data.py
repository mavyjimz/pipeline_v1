import pandas as pd
import os

def validate_schema():
    raw_path = 'input_data/raw/superstore_sales.csv'
    # The 'Truth' list we just found in image_fbceaa.png
    required_columns = ['Order Date', 'Ship Mode', 'Segment', 'Region', 'Category', 'Sales']
    
    if not os.path.exists(raw_path):
        print(f"CRITICAL: Raw data missing at {raw_path}")
        return False

    df = pd.read_csv(raw_path, encoding='latin1', nrows=5)
    actual_columns = df.columns.tolist()
    
    missing = [col for col in required_columns if col not in actual_columns]
    
    if missing:
        print(f"VALIDATION FAILED: Missing columns {missing}")
        return False
    
    print("VALIDATION SUCCESS: All required columns present.")
    return True

if __name__ == "__main__":
    validate_schema()