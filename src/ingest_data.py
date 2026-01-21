import os
import shutil
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def ingest_process():
    raw_dir = "../../input_data/raw/"
    # WE are explicitly looking for your superstore file now
    target_file = "superstore_sales.csv" 
    
    source_path = os.path.join(raw_dir, target_file)

    if os.path.exists(source_path):
        logging.info(f"SUCCESS: Found {target_file}. Ready for processing.")
        return True
    else:
        logging.error(f"FAIL: {target_file} not found in {raw_dir}")
        return False

if __name__ == "__main__":
    ingest_process()