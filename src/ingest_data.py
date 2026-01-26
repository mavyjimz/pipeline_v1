import os
import pandas as pd
import subprocess
import sys

# --- PATH CONFIGURATION ---
# Using absolute paths to ensure zero errors in D: drive architecture
MLOPS_ROOT = r"D:\MLOps"
RAW_FILE = os.path.join(MLOPS_ROOT, "input_data", "raw", "superstore_sales.csv")
# Path to the sanitizer script we built in Lesson 31
SANITIZER_SCRIPT = os.path.join(MLOPS_ROOT, "projects", "pipeline_v1", "src", "sanitizer.py")

def ingest():
    print("----------------------------------------------------")
    print("PHASE 7 LESSON 32: DATA INGESTION GATEWAY")
    print("----------------------------------------------------")

    # 1. CHECK FOR RAW DATA
    if not os.path.exists(RAW_FILE):
        print(f"ALERT: Raw file not found at {RAW_FILE}")
        print("ACTION REQUIRED: Ensure superstore_sales.csv is in the raw folder.")
        return False

    print("SUCCESS: Raw data detected in warehouse.")

    # 2. TRIGGER SANITIZATION (Lesson 31 Integration)
    # sys.executable ensures we use the current Virtual Environment's Python
    print("ACTION: Triggering Sanitizer via Environment Executable...")
    try:
        result = subprocess.run(
            [sys.executable, SANITIZER_SCRIPT], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            # Print the success message from the sanitizer script itself
            print(result.stdout)
            print("SUCCESS: Sanitization complete. Golden Path refreshed.")
            return True
        else:
            print("ERROR: Sanitization process failed.")
            print(f"DEBUG INFO: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"CRITICAL SYSTEM ERROR: {e}")
        return False

if __name__ == "__main__":
    ingest()