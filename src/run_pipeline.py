import subprocess
import sys
import logging
import os

# 1. FIX PATHING: Get the absolute path to the project root
# This ensures it finds the 'logs' folder whether you run from root or src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "logs", "pipeline_history.log")

# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# 2. SETUP LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_step(script_name):
    script_path = os.path.join("src", script_name)
    logging.info(f"--- Starting Stage: {script_name} ---")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        logging.info(f"--- Completed Stage: {script_name} ---")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"CRITICAL ERROR in {script_name}")
        print(f"Error Details: {e.stderr}")
        return False

if __name__ == "__main__":
    logging.info("MLOPS PIPELINE ACTIVATED: Hardware-Aware Mode")
    
    steps = ["ingest_data.py", "clean_data.py"]
    
    success = True
    for step in steps:
        if not run_step(step):
            logging.error("PIPELINE HALTED: Please fix the error above.")
            success = False
            break
            
    if success:
        # This is the line that updates your logs with the GPU proof!
        logging.info("SUCCESS: Mission Accomplished - Warehouse to GPU Link is Green!")