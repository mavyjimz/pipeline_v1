import os
import subprocess
import logging
import sys

# Setup Logging to track our RX 580 performance and pipeline health
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_worker(script_name):
    """Executes a pipeline stage and handles errors immediately."""
    script_path = os.path.join("src", script_name)
    logging.info(f"🚀 Starting Stage: {script_name}")
    
    try:
        # We use sys.executable to ensure we use your Drive D: Python environment
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
        logging.info(f"✅ Completed: {script_name}")
        # Print the inner script's output (like our 25-feature count)
        print(result.stdout) 
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ FAILED: {script_name}")
        logging.error(f"Error Details: {e.stderr}")
        sys.exit(1) # Stop the factory if one part breaks

def main():
    logging.info("================================================")
    logging.info("🏭 FACTORY START: PHASE IV -> PHASE V BRIDGE")
    logging.info("================================================")

    # STEP 1: Ingest Raw Data (Getting the Superstore CSV)
    run_worker("ingest_data.py")

    # STEP 2: Feature Engineering (The Lesson 19 Breakthrough)
    # This turns messy rows into 25 mathematical features!
    run_worker("clean_data.py")

    # STEP 3: AI Training (The Lesson 20 Goal)
    # This sends our 25 features to the RX 580 [cite: 2026-01-09]
    run_worker("train_model.py")

    logging.info("================================================")
    logging.info("🎯 MISSION ACCOMPLISHED: PIPELINE DEPLOYED")
    logging.info("================================================")

if __name__ == "__main__":
    main()