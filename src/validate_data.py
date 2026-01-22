import os
import pandas as pd
import glob

# UPDATED: Pointing to your actual RAW data warehouse
INPUT_DIR = r'D:\MLOps\input_data\raw'

def validate():
    print("\n--- 🛡️ DATA BOUNCER: PRE-FLIGHT CHECK ---")
    
    # Check if the directory even exists first
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ERROR: Directory not found: {INPUT_DIR}")
        return False
    
    # Find all CSV files in the raw folder
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    if not csv_files:
        print(f"❌ REJECTED: No CSV files found in {INPUT_DIR}")
        return False
    
    # Identify the newest file (likely superstore_sales.csv or test_sales.csv)
    latest_file = max(csv_files, key=os.path.getctime)
    filename = os.path.basename(latest_file)
    print(f"🔍 Checking latest file: {filename}")

    try:
        df = pd.read_csv(latest_file)
        
        # Validation Logic
        required_cols = ['Sales', 'Profit']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"❌ REJECTED: Missing required columns: {missing}")
            # Extra Tip: Check if column names are Case Sensitive (e.g., 'sales' vs 'Sales')
            return False
        
        if df.empty:
            print("❌ REJECTED: File is empty.")
            return False

        print(f"✅ PASSED: {len(df)} rows detected in {filename}.")
        print("🚀 READY FOR PIPELINE EXECUTION.")
        return True

    except Exception as e:
        print(f"❌ CRITICAL ERROR reading file: {e}")
        return False

if __name__ == "__main__":
    validate()