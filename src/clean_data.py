import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_FILE = r"D:\MLOps\input_data\raw\superstore_sales.csv"
OUTPUT_FOLDER = r"D:\MLOps\input_data\processed"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "cleaned_sales.csv")

def clean_data():
    print(f"🚀 Starting the Cleaning Station...")
    
    # 1. Load the data
    df = pd.read_csv(INPUT_FILE)
    
    # --- POWER WASH COLUMN NAMES (Mystery Solver) ---
    df.columns = [c.strip() for c in df.columns]
    print(f"🔍 The columns I found are: {df.columns.tolist()}")
    
    # 2. Fix the Dates (dayfirst handles the PH/Euro format)
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
    
    # 3. Handle Missing Values
    df['Postal Code'] = df['Postal Code'].fillna(0).astype(int)
    
    # 4. Data Enrichment (The Safety Guard)
    if 'Profit' in df.columns and 'Sales' in df.columns:
        print("💎 Adding Profit Margin...")
        df['Profit Margin'] = df['Profit'] / df['Sales']
    else:
        print("⚠️ Warning: Could not find 'Profit' or 'Sales' in the list above!")
    
    # 5. Save the Clean Version
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Success! Cleaned data saved to: {OUTPUT_FILE}")
    print(f"📊 Summary: Processed {len(df)} rows.")

if __name__ == "__main__":
    clean_data()