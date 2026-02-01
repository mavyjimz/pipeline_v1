import os
import pandas as pd
import glob

# 1. THE SETTINGS (The Warehouse Path)
INPUT_DIR = r'D:\MLOps\input_data\raw'
LOGBOOK_FILE = "detected_error.txt"

def validate():
    print("\n--- DATA BOUNCER: STARTING INSPECTION ---")
    
    # 2. CHECK IF FOLDER EXISTS
    if not os.path.exists(INPUT_DIR):
        error_msg = f"ERROR: Warehouse path not found: {INPUT_DIR}"
        print(error_msg)
        with open(LOGBOOK_FILE, "w") as f:
            f.write(error_msg)
        return False

    # 3. LOOK FOR TRUCKS (CSV FILES)
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        error_msg = "REJECTED: No CSV files found in warehouse!"
        print(error_msg)
        with open(LOGBOOK_FILE, "w") as f:
            f.write(error_msg)
        return False

    # 4. INSPECT EACH TRUCK
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"\nChecking: {filename}")

        try:
            df = pd.read_csv(file_path)
            cols = [c.lower() for c in df.columns]

            if 'sales' in cols:
                success_msg = f"RESULT: PASSED ({len(df)} rows found)"
                print(success_msg)
                # Clear the logbook if things are good
                with open(LOGBOOK_FILE, "w") as f:
                    f.write("SYSTEM HEALTHY: All Sales Data Verified.")
            else:
                error_msg = f"ALARM: Missing 'Sales' column in {filename}!"
                print(f"RESULT: FAILED ({error_msg})")
                with open(LOGBOOK_FILE, "w") as f:
                    f.write(error_msg)

        except Exception as e:
            error_msg = f"RESULT: ERROR reading file: {e}"
            print(error_msg)
            with open(LOGBOOK_FILE, "w") as f:
                f.write(error_msg)

    print("\n--- END OF REPORT ---")
    return True

if __name__ == "__main__":
    validate()