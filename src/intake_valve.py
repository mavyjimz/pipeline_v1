import os
import pandas as pd
import time

def scan_input_folder(folder_path):
    print(f"--- Scanning Intake Valve: {folder_path} ---")
    
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Filter for only CSV or Excel files
    data_files = [f for f in files if f.endswith('.csv') or f.endswith('.xlsx')]
    
    if not data_files:
        print("⚠️ No data files found. Please drop a CSV or XLSX in the folder.")
        return []
    
    print(f"✅ Found {len(data_files)} file(s) ready for processing:")
    for file in data_files:
        print(f"   - {file}")
    
    return data_files

if __name__ == "__main__":
    # Correcting the path to match your folder: input_data/raw
    # We point to 'raw' because we don't want to scan the 'processed' folder again!
    INPUT_DIR = "input_data/raw" 
    scan_input_folder(INPUT_DIR)