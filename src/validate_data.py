import os
import pandas as pd
import glob

# Your synchronized data warehouse path
INPUT_DIR = r'D:\MLOps\input_data\raw'

def validate():
    print("\n--- DATA BOUNCER: DIAGNOSTIC REPORT ---")
    
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Warehouse path not found: {INPUT_DIR}")
        return False
    
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print(f"REJECTED: No CSV files found in {INPUT_DIR}")
        return False
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"\nChecking: {filename}")
        
        try:
            df = pd.read_csv(file_path)
            # Case-insensitive column check
            cols = [c.lower() for c in df.columns]
            
            if 'sales' in cols:
                print(f"  RESULT: PASSED ({len(df)} rows found)")
            else:
                print(f"  RESULT: FAILED (Missing 'Sales' column)")
        except Exception as e:
            print(f"  RESULT: ERROR reading file: {e}")

    print("\n--- END OF REPORT ---")
    return True

if __name__ == "__main__":
    validate()