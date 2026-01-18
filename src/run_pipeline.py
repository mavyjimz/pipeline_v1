import subprocess
import os
import sys
from datetime import datetime

# Get the path where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# New: Define where the master log lives
LOG_FILE = r"D:\MLOps\logs\pipeline_history.log"

def log_event(message):
    """Prints to terminal and writes to a permanent log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    
    print(entry) # Keep the live feedback
    
    # "a" means append - it keeps the old history and adds to the bottom
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    log_event(f"🎬 STARTING: {script_name}...")
    
    # Run the script
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode == 0:
        log_event(f"✅ {script_name} finished successfully!")
    else:
        log_event(f"❌ ERROR in {script_name}. Pipeline stopped.")
        sys.exit(1)

if __name__ == "__main__":
    log_event("🚀 MLOPS PIPELINE ACTIVATED")
    log_event("============================")
    
    run_script("ingest_data.py")
    run_script("clean_data.py")
    run_script("visualize_data.py")
    run_script("train_model.py")
        
    log_event("🏆 MISSION ACCOMPLISHED: Factory is Green!")
    log_event("-" * 40) # A separator for the next time you run it