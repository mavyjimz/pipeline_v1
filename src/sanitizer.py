import pandas as pd
import os

# --- PATH CONFIGURATION ---
RAW_DATA_PATH = r'D:\MLOps\input_data\raw\superstore_sales.csv'
PROCESSED_DIR = r'D:\MLOps\input_data\processed'
CLEANED_DATA_PATH = os.path.join(PROCESSED_DIR, 'cleaned_sales.csv')

def run_sanitization_pipeline():
    print("----------------------------------------------------")
    print("PHASE 7 LESSON 33: TEMPORAL FEATURE ENGINEERING")
    print("----------------------------------------------------")

    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: File not found at {RAW_DATA_PATH}")
        return

    # 1. LOAD DATA
    df = pd.read_csv(RAW_DATA_PATH)

    # 2. THE SANITIZER (From Lesson 31)
    # Handle Postal Code and fill general NaNs
    df['Postal Code'] = df['Postal Code'].fillna(0).astype(int).astype(str)
    df = df.fillna(0)

    # 3. THE TEMPORAL ATTACK (New Lesson 33 Logic)
    print("STEP 1: EXTRACTING TEMPORAL FEATURES...")
    
    # Convert Order Date to datetime objects
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    
    # Extract mathematical signals
    df['Order_Month'] = df['Order Date'].dt.month
    df['Order_DayOfWeek'] = df['Order Date'].dt.dayofweek
    df['Order_Year'] = df['Order Date'].dt.year

    print(f"SUCCESS: Created Month and DayOfWeek features.")

    # 4. EGRESS
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"OUTPUT SAVED: {CLEANED_DATA_PATH}")
    print("----------------------------------------------------")

if __name__ == "__main__":
    run_sanitization_pipeline()