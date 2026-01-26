import os
import pandas as pd
import subprocess

# --- PATH CONFIGURATION ---
MLOPS_ROOT = r"D:\MLOps"
RAW_FILE = os.path.join(MLOPS_ROOT, "input_data", "raw", "superstore_sales.csv")
SANITIZER_SCRIPT = os.path.join(MLOPS_ROOT, "projects", "pipeline_v1", "src", "sanitizer.py")

def ingest():
    print("----------------------------------------------------")
    print("PHASE 7 LESSON 32: DATA INGESTION GATEWAY")
    print("----------------------------------------------------")

    # 1. CHECK FOR RAW DATA
    if not os.path.exists(RAW_FILE):
        print(f"ALERT: Raw file not found at {RAW_FILE}")
        print("Please ensure superstore_sales.csv is in the raw folder.")
        return False

    print("SUCCESS: Raw data detected in the official warehouse.")

    # 2. TRIGGER SANITIZATION (Lesson 31 Integration)
    print("ACTION: Triggering Sanitizer to refresh Golden Path...")
    try:
        # This runs your sanitizer.py automatically
        result = subprocess.run(["python", SANITIZER_SCRIPT], capture_output=True, text=True)
        if result.returncode == 0:
            print("SUCCESS: Sanitization complete. Golden Path is ready.")
            return True
        else:
            print("ERROR: Sanitizer failed. Check sanitizer.py logic.")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"SYSTEM ERROR during sanitization: {e}")
        return False

if __name__ == "__main__":
    ingest()