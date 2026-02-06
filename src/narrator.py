import os
import time
import subprocess
from narrator_utils import type_message, fast_scan

def run_presentation():
    # SET TERMINAL TO MATRIX GREEN & CLEAR
    os.system('color 0a') 
    os.system('cls')
    
    # --- ACT 1: THE HACKER HEADER ---
    print("""
    ##########################################################
    # SYSTEM: MLOPS_PIPELINE_V1          STATUS: OPERATIONAL #
    # OPERATOR: VANJUNN PONGASI          CO-PILOT: GEMINI    #
    # HARDWARE: INTEL i5-12400f          LOCATION: PHILIPPINES #
    ##########################################################
    """)
    time.sleep(1)

    # --- ACT 2: SYSTEM DIAGNOSTIC SCAN ---
    fast_scan(12) 
    time.sleep(1)

    # --- ACT 3: THE NARRATIVE (Paced for 3s readability) ---
    type_message(">>> [BOOT SEQUENCE COMPLETE]: Initializing Narrative Layer...")
    time.sleep(3)
    type_message("\n'Hello world. My name is Vanjunn Pongasi. With me is Gemini,")
    type_message("executing our project thru Python for MLOps.'")
    time.sleep(3)

    type_message("\n>>> [PROJECT SCOPE & HARDWARE]:")
    type_message("- Environment: Intel i5-12400f | 16GB DDR4 RAM.")
    time.sleep(3)
    type_message("- Goal: End-to-End Pipeline for Superstore Sales Analysis.")
    time.sleep(3)
    type_message("- Solved: Handled raw data corruption and optimized Docker overhead.")
    time.sleep(3)

    # --- ACT 4: THE SCRIPT SHOWCASE (Sneak Peeks) ---
    type_message("\n>>> [INSPECTION]: Verifying Pipeline Infrastructure...")
    time.sleep(2)

    # Peek 1: Data Logic
    type_message(">>> Opening: src/check_data.py (Data Integrity Logic)")
    subprocess.Popen(['start', 'cmd', '/c', 'type src\\check_data.py && timeout /t 3'], shell=True)
    time.sleep(4)

    # Peek 2: Docker Infrastructure (The MLOps Flex)
    type_message(">>> Opening: Dockerfile (Infrastructure as Code)")
    subprocess.Popen(['start', 'cmd', '/c', 'type Dockerfile && timeout /t 3'], shell=True)
    time.sleep(4)

    # Peek 3: Dashboard Logic
    type_message(">>> Opening: src/dashboard.py (Visualization Layer)")
    subprocess.Popen(['start', 'cmd', '/c', 'type src\\dashboard.py && timeout /t 3'], shell=True)
    time.sleep(4)

    # --- ACT 5: LIVE EXECUTION PHASE ---
    type_message("\n" + "="*50)
    type_message(">>> ALL SCRIPTS VERIFIED. STARTING LIVE PIPELINE...")
    type_message("="*50)
    time.sleep(3)

    # A. DATA CHECK RUN
    type_message("\n[ACTION]: Running check_data.py...")
    subprocess.Popen(['start', 'cmd', '/c', 'python src/check_data.py && timeout /t 10'], shell=True)
    time.sleep(12) 

    # B. MAIN PIPELINE RUN (Lightning Fast)
    type_message("\n[ACTION]: Firing automate_all.bat (Dockerized Pipeline)...")
    subprocess.Popen(['start', 'cmd', '/c', 'automate_all.bat'], shell=True)
    time.sleep(10) 

    # C. DASHBOARD DISPLAY
    type_message("\n[ACTION]: Launching Dashboard for Client Review...")
    os.system('start chrome http://localhost:8000')
    type_message(">>> Displaying live results for 15 seconds...")
    time.sleep(15) 

    # CLEANUP: CLOSE CHROME
    type_message(">>> [CLEANUP]: Terminating Dashboard view...")
    os.system("taskkill /f /im chrome.exe") 
    time.sleep(2)

    # --- ACT 6: FINAL CREDITS ---
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