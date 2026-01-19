import pandas as pd
import argparse
import os

def main():
    # 1. Setup the "Front Desk" (Arguments)
    parser = argparse.ArgumentParser(description="MLOps Data Guard - Initial CSV Check")
    parser.add_argument("--file", type=str, required=True, help="Path to the CSV file")
    args = parser.parse_args()

    FILE_PATH = args.file

    # 2. Security Check: Does the file even exist?
    if os.path.exists(FILE_PATH):
        try:
            # 3. Load the data
            df = pd.read_csv(FILE_PATH)
            print(f"✅ SUCCESS: Loaded {FILE_PATH}")
            print(f"📊 SHAPE: {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Show the first 5 rows
            print("\n--- DATA PREVIEW ---")
            print(df.head())

            # 4. Professional Health Check (The logic we added!)
            print("\n--- DATA HEALTH CHECK ---")
            null_counts = df.isnull().sum().sum()
            
            if null_counts > 0:
                print(f"⚠️  WARNING: Found {null_counts} missing values!")
            else:
                print("✅ CLEAN: No missing values found.")
            
            # Check if the file is too small
            if len(df) < 5:
                print("⚠️  CRITICAL: Dataset is very small. Check source!")

        except Exception as e:
            print(f"❌ ERROR: Could not read the file. Details: {e}")
    else:
        print(f"❌ ERROR: File not found at {FILE_PATH}")

if __name__ == "__main__":
    main()