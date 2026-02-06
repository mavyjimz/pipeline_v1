import os
import time
import subprocess
from narrator_utils import type_message, fast_scan

def run_presentation():
    # SET TERMINAL TO MATRIX GREEN
    os.system('color 0a') 
    os.system('cls')
    
    # --- STEP 2: THE HACKER HEADER ---
    print("""
    ##########################################################
    # SYSTEM: MLOPS_PIPELINE_V1          STATUS: OPERATIONAL #
    # OPERATOR: VANJUNN PONGASI          CO-PILOT: GEMINI    #
    # HARDWARE: INTEL i5-12400f          LOCATION: PHILIPPINES #
    ##########################################################
    """)
    time.sleep(1)

    # --- THE SYSTEM SCAN (Visual Noise) ---
    fast_scan(12) 
    time.sleep(1)

    # --- ACT 1: INTRO (Paced with 3s delays) ---
    type_message(">>> [BOOT SEQUENCE COMPLETE]: Initializing Narrative Layer...")
    time.sleep(3)
    type_message("\n'Hello world. My name is Vanjunn Pongasi. With me is Gemini,")
    type_message("executing our project thru Python for MLOps.'")
    time.sleep(3)

    # --- ACT 2: PROJECT DEPTH ---
    type_message("\n>>> [PROJECT SCOPE]:")
    type_message("- We built a full-scale MLOps Pipeline using 'superstore_sales.csv'.")
    time.sleep(3)
    type_message("- Accomplished: Automated Data Validation, Dockerization, and CI/CD.")
    time.sleep(3)
    type_message("- Solved: Handled data corruption, JSON schema mapping, and environment sync.")
    time.sleep(3)

    # --- ACT 3: HARDWARE CONSTRAINTS ---
    type_message("\n>>> [HARDWARE SIGNATURE]:")
    type_message("- Environment: Intel i5-12400f | 16GB DDR4 RAM.")
    time.sleep(3)
    type_message("- Challenge: Optimized Docker overhead for minimal hardware.")
    time.sleep(3)
    type_message("- Result: Proven efficiency; high performance without server costs.")
    time.sleep(3)

    # --- ACT 4: EXECUTION PHASE ---
    type_message("\n" + "="*50)
    type_message(">>> STARTING LIVE PIPELINE EXECUTION...")
    type_message("="*50)
    time.sleep(3)

    # A. DATA CHECK
    type_message("\n[ACTION]: Running check_data.py (Data Health Check)...")
    subprocess.Popen(['start', 'cmd', '/c', 'python src/check_data.py && timeout /t 10'], shell=True)
    time.sleep(12) 

    # B. MAIN PIPELINE
    type_message("\n[ACTION]: Firing automate_all.bat (Main MLOps Pipeline)...")
    subprocess.Popen(['start', 'cmd', '/c', 'automate_all.bat'], shell=True)
    time.sleep(10) 

    # C. DASHBOARD
    type_message("\n[ACTION]: Launching Dashboard for Client Review...")
    os.system('start chrome http://localhost:8000')
    type_message(">>> Displaying live results for 15 seconds...")
    time.sleep(15) 

    # CLEANUP: CLOSE CHROME
    type_message(">>> [CLEANUP]: Terminating Dashboard view...")
    os.system("taskkill /f /im chrome.exe") 
    time.sleep(2)

    # --- FINAL CREDITS ---
    os.system('cls') 
    print("\n" + "X"*50)
    type_message("MISSION ACCOMPLISHED")
    time.sleep(3)
    type_message("Project: End-to-End MLOps Pipeline")
    time.sleep(3)
    type_message("Lead Engineer: Vanjunn Pongasi")
    time.sleep(3)
    type_message("Co-Pilot: Gemini")
    time.sleep(3)
    type_message("Status: 100% Verified on i5-12400f")
    time.sleep(3)
    type_message("\n'Through floods and typhoons, the pipeline stands.'")
    print("X"*50)

if __name__ == "__main__":
    run_presentation()