import subprocess
import os
import sys
from datetime import datetime

# 1. Setup paths to be rock-solid
# This finds the 'src' folder regardless of where you run the command from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure the logs folder exists on your D: drive
LOG_DIR = r"D:\MLOps\logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, "pipeline_history.log")

def log_event(message):
    """Prints to terminal and writes to a permanent log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    
    # NEW: Safety check - does the file actually exist?
    if not os.path.exists(script_path):
        log_event(f"⚠️ SKIPPING: {script_name} not found at {script_path}")
        return

    log_event(f"🎬 STARTING: {script_name}...")
    
    # Run the script and wait for it to finish
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode == 0:
        log_event(f"✅ {script_name} finished successfully!")
    else:
        log_event(f"❌ ERROR in {script_name}. Pipeline stopped.")
        sys.exit(1)

if __name__ == "__main__":
    log_event("🚀 MLOPS PIPELINE ACTIVATED")
    log_event("================================")
    
    # The Sequence
    run_script("intake_valve.py")
    run_script("gpu_loader.py")
    run_script("speed_battle.py")
    
    log_event("================================")
    log_event("🏆 MISSION ACCOMPLISHED: Warehouse to GPU Link is Green!")