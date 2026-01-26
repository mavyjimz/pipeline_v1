import pandas as pd
import os

# --- PATH CONFIGURATION ---
# Using raw strings to ensure Windows backslashes are handled correctly
RAW_DATA_PATH = r'D:\MLOps\input_data\raw\superstore_sales.csv'
PROCESSED_DIR = r'D:\MLOps\input_data\processed'
CLEANED_DATA_PATH = os.path.join(PROCESSED_DIR, 'cleaned_sales.csv')

def run_sanitization_pipeline():
    print("----------------------------------------------------")
    print("PHASE 7 LESSON 31: DATA SANITIZATION STARTING")
    print("----------------------------------------------------")

    # 1. VERIFY PATHS
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: File not found at {RAW_DATA_PATH}")
        return

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        print(f"FOLDER CREATED: {PROCESSED_DIR}")

    # 2. THE SCOUT (Audit)
    print("STEP 1: AUDITING RAW DATA...")
    df = pd.read_csv(RAW_DATA_PATH)
    initial_nulls = df.isnull().sum().sum()
    print(f"TOTAL MISSING VALUES FOUND: {initial_nulls}")
    
    if initial_nulls > 0:
        print("COLUMNS WITH MISSING DATA:")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    # 3. THE SANITIZER (Cleaning)
    print("STEP 2: INITIATING FILL-PROTOCOL...")
    
    # Handle Postal Code: Fill NaNs with 0, convert to int, then to string
    # This prevents the float64 mismatch and keeps categories consistent
    df['Postal Code'] = df['Postal Code'].fillna(0).astype(int).astype(str)

    # 4. EGRESS (Saving)
    print("STEP 3: EXPORTING CLEANED DATA...")
    df.to_csv(CLEANED_DATA_PATH, index=False)
    
    # 5. FINAL VERIFICATION
    final_nulls = df.isnull().sum().sum()
    if final_nulls == 0:
        print("SUCCESS: Data is now 100 percent clean.")
        print(f"OUTPUT PATH: {CLEANED_DATA_PATH}")
    else:
        print(f"WARNING: {final_nulls} missing values still remain in the dataset.")

    print("----------------------------------------------------")

if __name__ == "__main__":
    run_sanitization_pipeline()