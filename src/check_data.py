import pandas as pd
import argparse
import os

# 1. Setup the "Front Desk" (Argparse)
parser = argparse.ArgumentParser(description="Professional Data Checker")
parser.add_argument("filename", help="The name of the CSV file inside the raw folder")

args = parser.parse_args()

# 2. Build the full path automatically
RAW_FOLDER = r"D:\MLOps\input_data\raw"
FILE_PATH = os.path.join(RAW_FOLDER, args.filename)

# 3. Check and Run
if os.path.exists(FILE_PATH):
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"\n✅ SUCCESSFULLY LOADED: {args.filename}")
        print(f"Total Rows: {len(df)} | Total Columns: {len(df.columns)}")
        print("\n--- FIRST 5 ROWS ---")
        print(df.head())
    except Exception as e:
        print(f"❌ ERROR: Could not read file. {e}")
else:
    print(f"❌ ERROR: File '{args.filename}' not found in {RAW_FOLDER}")