import os
import subprocess
import sys
import pandas as pd
from pathlib import Path

# --- DYNAMIC PATH CONFIGURATION ---
# BASE_DIR automatically finds the project root on Windows or Linux
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "input_data" / "raw" / "superstore_sales.csv"
SANITIZER_SCRIPT = BASE_DIR / "src" / "sanitizer.py"
PROCESSED_FILE = BASE_DIR / "input_data" / "processed" / "cleaned_sales.csv"

def ingest_data():
    print("--- PHASE 2: Starting Industrial Data Ingestion ---")
    print(f"Directory Root: {BASE_DIR}")
    print(f"Targeting File: {RAW_FILE}")

    if not RAW_FILE.exists():
        print(f"ERROR: Raw file not found at {RAW_FILE}")
        print("ACTION: Move superstore_sales.csv to the correct input_data/raw folder.")
        return

    print("SUCCESS: Raw file detected. Triggering Sanitizer script...")

    try:
        # STEP 1: Execute Sanitizer with UTF-8 to handle special characters
        result = subprocess.run(
            [sys.executable, str(SANITIZER_SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print("--- SANITIZER LOGS ---")
        print(result.stdout)

        # STEP 2: Load the data for Temporal Feature Engineering
        # Using encoding_errors='ignore' to strip emojis during the read process
        df = pd.read_csv(RAW_FILE, encoding='utf-8', encoding_errors='ignore', low_memory=False)

        print("--- PHASE 2: Temporal Feature Engineering ---")
        # Standardize Order Date
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        
        # Extract features for the portfolio model
        df['Order_Month'] = df['Order Date'].dt.month
        df['Order_DayOfWeek'] = df['Order Date'].dt.dayofweek
        
        print(f"SUCCESS: Extracted Month and DayOfWeek from {len(df)} rows.")

        # STEP 3: Save the final healthy file
        os.makedirs(PROCESSED_FILE.parent, exist_ok=True)
        df.to_csv(PROCESSED_FILE, index=False)
        print(f"PIPELINE OUTPUT: {PROCESSED_FILE}")
        print("--- PHASE 2: Process Complete. ---")

    except subprocess.CalledProcessError as e:
        print("ERROR: Sanitizer execution failed!")
        print(f"Details: {e.stderr}")
    except Exception as e:
        print(f"CRITICAL SYSTEM ERROR: {str(e)}")

if __name__ == "__main__":
    ingest_data()