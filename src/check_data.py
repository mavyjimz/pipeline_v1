import pandas as pd
import os

def scout_raw_data():
    # Use the source you confirmed: superstore_sales.csv
    file_path = r"D:\MLOps\input_data\raw\superstore_sales.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)

    print("\n--- 🛡️ LEAD'S RAW DATA RECON ---")
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns Found: {len(df.columns)}")
    
    print("\n--- 📑 ALL DETECTED HEADERS ---")
    print(df.columns.tolist())
    
    print("\n--- 📊 DATA TYPES (The 'Hidden' Categories) ---")
    print(df.dtypes)

if __name__ == "__main__":
    scout_raw_data()