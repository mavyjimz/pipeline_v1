import os
import subprocess
import logging
import sys

# 1. SETUP: Absolute Warehouse Path
LOG_FILE = r"D:\MLOps\logs\pipeline.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Determine the absolute directory where this run_pipeline.py is located
# This prevents the "src/src/" double-path ghost.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_worker(script_name):
    """Executes a pipeline stage using absolute paths."""
    # This combines the folder of this script with the script name
    script_path = os.path.join(BASE_DIR, script_name)
    logging.info(f"--- [STAGE START]: {script_name} ---")

    try:
        # Use sys.executable to ensure we stay in your 'venv'
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        logging.info(f"--- [STAGE SUCCESS]: {script_name} ---")
        if result.stdout:
            print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else "No error message captured."
        logging.error(f"--- [STAGE FAILED]: {script_name} ---")
        logging.error(f"Worker Error Output: {error_msg}")
        sys.exit(1)

def main():
    logging.info("=============================================")
    logging.info("START: Pipeline Orchestrator Initialized...")
    logging.info("=============================================")

    # List only the filenames; BASE_DIR handles the 'src/' part automatically
    scripts = [
        "ingest_data.py", 
        "clean_data.py", 
        "train_model.py"
    ]
    
    for script in scripts:
        run_worker(script)

    logging.info("DONE: Mission Accomplished. Pipeline Cycle Complete.")

if __name__ == "__main__":
    main()