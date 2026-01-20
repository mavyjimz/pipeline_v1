import pandas as pd
import os

def check_raw_data():
    # REVISED PATH: Pointing to your D: drive Warehouse
    file_path = r'D:\MLOps\input_data\raw\test_data.csv'
    
    print(f"🔍 Intake Valve: Checking {file_path}")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print("✅ Data Found!")
        print(f"📊 Summary: {len(df)} rows and {len(df.columns)} columns detected.")
        print("📝 First 5 rows:")
        print(df.head())
    else:
        print(f"❌ Error: Could not find the file at {file_path}")

if __name__ == "__main__":
    check_raw_data()