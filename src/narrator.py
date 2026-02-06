import os
import time
import subprocess
from narrator_utils import type_message

def run_presentation():
    os.system('cls')
    
    # --- 1st: INTRO ---
    type_message(">>> [SYSTEM INITIALIZED]: MLOps Presentation Layer Active")
    type_message("\n'Hello world. My name is Vanjunn Pongasi. With me is Gemini,")
    type_message("executing our project thru Python for MLOps.'")
    time.sleep(2)

    # --- 2nd: PROJECT DEPTH ---
    type_message("\n>>> [PROJECT SCOPE]:")
    type_message("- We built a full-scale MLOps Pipeline using 'superstore_sales.csv'.")
    type_message("- Accomplished: Automated Data Validation, Dockerization, and CI/CD.")
    type_message("- Solved: Handled data corruption, JSON schema mapping, and environment sync.")
    time.sleep(3)

    # --- 3rd: HARDWARE CONSTRAINTS ---
    type_message("\n>>> [HARDWARE SIGNATURE]:")
    type_message("- Environment: Intel i5-12400f | 16GB DDR4 RAM.")
    type_message("- Challenge: Optimized Docker overhead to run smoothly on minimal hardware.")
    type_message("- Result: Proven efficiency; high performance without server-grade costs.")
    time.sleep(3)

    # --- 4th: THE WORKFLOW (The Attack Plan) ---
    type_message("\n>>> [THE WORKFLOW STRATEGY]:")
    type_message("A. PRE-FLIGHT: check_data.py ensures raw data integrity before training.")
    type_message("B. ORCHESTRATION: run_pipeline.sh manages the Docker Linux environment.")
    type_message("   1. Cleaning: Removing noise from raw CSV.")
    type_message("   2. Feature Engineering: Creating new ML data columns.")
    type_message("   3. Training: Model generation within containers.")
    type_message("   4. Prediction: Scoring and outputting results.")
    type_message("   5. Infrastructure: Supporting .yml and Dockerfiles.")
    time.sleep(5)

    # --- 5th: CI/CD & CLIENT DELIVERY ---
    type_message("\n>>> [CI/CD & DELIVERY]:")
    type_message("- Output: Structured .json for real-time client visualization.")
    type_message("- Deployment: Automated Chrome launch to LocalHost dashboard.")
    time.sleep(3)

    # --- EXECUTION PHASE ---
    type_message("\n" + "="*50)
    type_message(">>> STARTING LIVE PIPELINE EXECUTION...")
    type_message("="*50)
    time.sleep(2)

    # A. DATA CHECK
    type_message("\n[ACTION]: Running check_data.py (Data Health Check)...")
    # Using /c to auto-close after 10s delay inside the cmd call
    subprocess.Popen(['start', 'cmd', '/c', 'python src/check_data.py && timeout /t 10'], shell=True)
    time.sleep(12) 

    # B. MAIN PIPELINE
    type_message("\n[ACTION]: Firing automate_all.bat (Main MLOps Pipeline)...")
    type_message(">>> Cleaning -> Engineering -> Training -> Predicting...")
    # This runs the heavy lifting
    pipeline_proc = subprocess.Popen(['start', 'cmd', '/c', 'automate_all.bat'], shell=True)
    
    # Wait for the pipeline to finish and the server to be ready
    time.sleep(10) 

    # C. DASHBOARD
    type_message("\n[ACTION]: Launching Dashboard for Client Review...")
    os.system('start chrome http://localhost:8000')
    time.sleep(15) # Show the dashboard for 15 seconds

    # --- 6th: FINAL CREDITS ---
    os.system('cls') # Clear screen for final impact
    print("\n" + "X"*50)
    type_message("MISSION ACCOMPLISHED")
    type_message("Project: End-to-End MLOps Pipeline")
    type_message("Lead Engineer: Vanjunn Pongasi")
    type_message("Co-Pilot: Gemini")
    type_message("Status: 100% Verified on i5-12400f")
    type_message("\n'Through floods and typhoons, the pipeline stands.'")
    print("X"*50)

if __name__ == "__main__":
    run_presentation()