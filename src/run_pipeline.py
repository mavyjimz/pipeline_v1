import subprocess
import sys
import logging
import os

# --- 1. SYNCED PATHING (MLOps ROOT) ---
# SCRIPT_DIR is MLOps/projects/pipeline_v1/src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Move up THREE levels to get to D:/MLOps
# src -> pipeline_v1 -> projects -> MLOps
MLOPS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

LOG_FOLDER = os.path.join(MLOPS_ROOT, "logs")
LOG_FILE = os.path.join(LOG_FOLDER, "pipeline_history.log")

os.makedirs(LOG_FOLDER, exist_ok=True)

# --- 2. LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_worker(file_name):
    worker_path = os.path.join(SCRIPT_DIR, file_name)
    logging.info(f"--- EXECUTING: {file_name} ---")
    try:
        result = subprocess.run([sys.executable, worker_path], capture_output=True, text=True, check=True)
        logging.info(result.stdout)
        logging.info(f"--- SUCCESS: {file_name} finished ---")
    except subprocess.CalledProcessError as e:
        logging.error(f"!!! CRASH IN {file_name} !!!\n{e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    logging.info("PIPELINE START: Hardware Check - AMD RX 580")
    
    run_worker("ingest_data.py")
    run_worker("clean_data.py")

    # This is the line you were missing!
    logging.info("SUCCESS: Mission Accomplished - Warehouse to GPU Link is Green!")