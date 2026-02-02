import os
import subprocess
import sys
from pathlib import Path

# DYNAMIC PATH CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "input_data" / "raw" / "superstore_sales.csv"
SANITIZER_SCRIPT = BASE_DIR / "src" / "sanitizer.py"

def ingest_data():
    print("PHASE 2: Starting Data Ingestion...")
    print(f"Directory: {BASE_DIR}")
    print(f"Target File: {RAW_FILE}")

    if not RAW_FILE.exists():
        print(f"ERROR: Raw file not found at {RAW_FILE}")
        print("ACTION: Ensure superstore_sales.csv is in the correct folder.")
        return

    print("SUCCESS: Raw file detected. Triggering Sanitizer...")

    try:
        # Using utf-8 encoding to prevent crash during handoff
        result = subprocess.run(
            [sys.executable, str(SANITIZER_SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print("--- SANITIZER OUTPUT ---")
        print(result.stdout)
        print("PHASE 2: Process Complete.")
        
    except subprocess.CalledProcessError as e:
        print("ERROR: Sanitizer execution failed.")
        print(f"Details: {e.stderr}")
    except Exception as e:
        print(f"CRITICAL SYSTEM ERROR: {str(e)}")

if __name__ == "__main__":
    ingest_data()