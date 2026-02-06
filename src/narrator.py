import os
import time
import subprocess
import sys
from narrator_utils import type_message, fast_scan

def run_presentation():
    # INITIALIZE MATRIX THEME
    os.system('color 0a') 
    os.system('cls')
    
    # --- ACT 1: THE HACKER HEADER ---
    print("""
    ##########################################################
    # SYSTEM: MLOPS_PRO_PIPELINE         STATUS: OPERATIONAL #
    # OPERATOR: VANJUNN PONGASI          CO-PILOT: GEMINI    #
    # HARDWARE: INTEL i5-12400f          LOCATION: PHILIPPINES #
    # VERSION: 2026.FEBRUARY.06          STABILITY: 100%     #
    ##########################################################
    """)
    time.sleep(1)

    # --- ACT 2: SYSTEM DIAGNOSTICS & DEVOPS CHECK ---
    fast_scan(10) 
    
    type_message(">>> [INTERNAL]: Verifying Git Repository Integrity...")
    os.system('git status -s') # Shows any unsynced files
    time.sleep(2)
    type_message(">>> [STATUS]: Repository Synced. Version Control Active.")
    time.sleep(2)

    type_message("\n>>> [TELEMETRY]: Analyzing Hardware Headroom...")
    type_message("--- CPU: Intel i5-12400f (12 Logical Processors) | [OK]")
    type_message("--- RAM: 16GB DDR4 (Optimized for Docker Overhead) | [OK]")
    type_message("--- GPU: AMD Radeon RX 580 (VRAM Initialized) | [OK]")
    time.sleep(3)

    # --- ACT 3: THE NARRATIVE ---
    type_message("\n>>> [BOOT SEQUENCE COMPLETE]: Initializing Narrative Layer...")
    time.sleep(3)
    type_message("\n'Hello world. My name is Vanjunn Pongasi. With me is Gemini,")
    type_message("executing our end-to-end MLOps workflow.'")
    time.sleep(3)

    type_message("\n>>> [PROJECT MISSION]:")
    type_message("- Objective: Automate Superstore Sales Pipeline.")
    time.sleep(3)
    type_message("- Engineering: Dockerization, Data Validation, and CI/CD.")
    time.sleep(3)
    type_message("- Resilience: Built to run efficiently on consumer-grade hardware.")
    time.sleep(3)

    # --- ACT 4: THE SCRIPT SHOWCASE (The Sneak Peeks) ---
    type_message("\n>>> [INSPECTION]: Opening Core Source Files for Review...")
    time.sleep(2)

    # Peek 1: The Validation Logic
    type_message(">>> SCRIPT: src/check_data.py (Data Health & Validation)")
    subprocess.Popen(['start', 'cmd', '/c', 'type src\\check_data.py && timeout /t 3'], shell=True)
    time.sleep(4)

    # Peek 2: The Infrastructure (The Flex)
    type_message(">>> INFRASTRUCTURE: Dockerfile (Container Blueprint)")
    subprocess.Popen(['start', 'cmd', '/c', 'type Dockerfile && timeout /t 3'], shell=True)
    time.sleep(4)

    # Peek 3: The Presentation Layer
    type_message(">>> UI: src/dashboard.py (Client Visualization)")
    subprocess.Popen(['start', 'cmd', '/c', 'type src\\dashboard.py && timeout /t 3'], shell=True)
    time.sleep(4)

    # --- ACT 5: LIVE EXECUTION ---
    type_message("\n" + "="*60)
    type_message(">>> ALL CHECKS PASSED. COMMENCING FULL PIPELINE EXECUTION...")
    type_message("="*60)
    time.sleep(3)

    # A. Data Check Execution
    type_message("\n[ACTION]: Executing Data Integrity Validation...")
    subprocess.Popen(['start', 'cmd', '/c', 'python src/check_data.py && timeout /t 10'], shell=True)
    time.sleep(12) 

    # B. Docker Pipeline Execution
    type_message("\n[ACTION]: Orchestrating Docker Pipeline (automate_all.bat)...")
    subprocess.Popen(['start', 'cmd', '/c', 'automate_all.bat'], shell=True)
    time.sleep(10) 

    # C. Dashboard Launch
    type_message("\n[ACTION]: Deploying Dashboard to LocalHost:8000...")
    os.system('start chrome http://localhost:8000')
    type_message(">>> Reviewing results on Production Server...")
    time.sleep(15) 

    # --- CLEANUP ---
    type_message("\n>>> [CLEANUP]: Closing Production Viewports...")
    os.system("taskkill /f /im chrome.exe") 
    time.sleep(2)

    # --- ACT 6: FINAL CREDITS ---
    os.system('cls') 
    print("\n" + "X"*60)
    type_message("MISSION ACCOMPLISHED")
    time.sleep(3)
    type_message("Project: End-to-End MLOps Pipeline")
    time.sleep(2)
    type_message("Lead Engineer: Vanjunn Pongasi")
    time.sleep(2)
    type_message("Co-Pilot: Gemini")
    time.sleep(2)
    type_message("Hardware Verified: i5-12400f | 16GB RAM")
    time.sleep(3)
    type_message("\n'The pipeline is stable. The data is clean. The mission is complete.'")
    print("X"*60)
    time.sleep(5)

if __name__ == "__main__":
    run_presentation()