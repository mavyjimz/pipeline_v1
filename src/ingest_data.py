import os
import shutil
import logging
from datetime import datetime

# --- CONFIGURATION ---
# Updated absolute paths for your new project structure
DOWNLOADS_PATH = r"C:\Users\mavy\Documents\Downloads"
DESTINATION_PATH = r"D:\MLOps\input_data\raw"
LOG_FILE = r"D:\MLOps\logs\ingestion.log"

# Setup Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ingest_files():
    print("Checking for new data...")
    # Supported file extensions
    extensions = ('.csv', '.xlsx', '.xls')
    
    files_moved = 0
    
    # Scan the downloads folder
    for filename in os.listdir(DOWNLOADS_PATH):
        if filename.lower().endswith(extensions):
            source = os.path.join(DOWNLOADS_PATH, filename)
            destination = os.path.join(DESTINATION_PATH, filename)
            
            try:
                # Move the file
                shutil.move(source, destination)
                logging.info(f"SUCCESS: Moved {filename} to {DESTINATION_PATH}")
                print(f"Successfully moved: {filename}")
                files_moved += 1
            except Exception as e:
                logging.error(f"ERROR: Could not move {filename}. Reason: {e}")
                print(f"Error moving {filename}. Check logs.")

    if files_moved == 0:
        print("No new CSV or Excel files found.")
    else:
        print(f"Done! {files_moved} files are now in your warehouse.")

if __name__ == "__main__":
    ingest_files()