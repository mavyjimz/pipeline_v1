import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="MLOps Data Scout - Column Auditor")
    parser.add_argument("--file", type=str, required=True, help="Path to the CSV file")
    args = parser.parse_args()

    if os.path.exists(args.file):
        df = pd.read_csv(args.file)
        
        print(f"\n✅ SUCCESS: Loaded {args.file}")
        print(f"📊 SHAPE: {df.shape[0]} rows and {df.shape[1]} columns")
        
        # --- THE COLUMN AUDIT ---
        print("\n🔍 FULL COLUMN LIST:")
        all_columns = df.columns.tolist()
        for i, col in enumerate(all_columns, 1):
            print(f"{i}. {col}")

        # --- THE SAFETY CHECK ---
        target_cols = ['Category', 'Sales'] # Add 'Profit' here if you want to test
        print("\n🛡️ SAFETY CHECK:")
        for col in target_cols:
            if col in all_columns:
                print(f"✅ FOUND: '{col}' is present.")
            else:
                print(f"❌ MISSING: '{col}' NOT FOUND! Pipeline will crash.")

    else:
        print(f"❌ ERROR: File not found at {args.file}")

if __name__ == "__main__":
    main()