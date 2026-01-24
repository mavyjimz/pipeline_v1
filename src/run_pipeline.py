import os
import subprocess
import logging
import sys

# Hardened Logging: Forced UTF-8 encoding for the file to prevent charmap crashes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_worker(script_name):
    """Executes a pipeline stage with high-compatibility text handling."""
    script_path = os.path.join("src", script_name)
    logging.info(f"[STAGE START]: {script_name}")
    
    try:
        # sys.executable ensures we stay in your Drive D: virtual environment
        result = subprocess.run(
            [sys.executable, script_path], 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8' # Force UTF-8 capture from the worker scripts
        )
        logging.info(f"[STAGE SUCCESS]: {script_name}")
        if result.stdout:
            print(result.stdout.strip()) 
    except subprocess.CalledProcessError as e:
        logging.error(f"[STAGE FAILED]: {script_name}")
        # Capture the actual error from the worker for the log
        error_msg = e.stderr if e.stderr else "No error message captured."
        logging.error(f"Worker Error Output: {error_msg}")
        sys.exit(1)

def main():
    logging.info("================================================")
    logging.info("FACTORY START: PHASE IV -> PHASE V BRIDGE")
    logging.info("================================================")

    # Step 1: Ingest (Warehouse to Project)
    run_worker("ingest_data.py")

    # Step 2: Clean (The 25-Feature Expansion)
    run_worker("clean_data.py")

    # Step 3: Train (Lesson 20 AI Engine)
    run_worker("train_model.py")

    logging.info("================================================")
    logging.info("MISSION ACCOMPLISHED: PIPELINE DEPLOYED")
    logging.info("================================================")

if __name__ == "__main__":
    main()