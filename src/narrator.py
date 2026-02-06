import os
import time
import subprocess
from narrator_utils import type_message

def run_presentation():
    os.system('cls')
    
    # ACT 1: INTRO (matches your schematic)
    type_message(">>> [BOOTING]: MLOps Presentation Layer...")
    type_message(">>> [PC SPECS]: Intel i5-12400f | RX 580 Active")
    type_message("\n'Hello world my name is Vanjunn Pongasi with me is Gemini")
    type_message("executing our project thru python for MLOps'")
    time.sleep(2)

    # ACT 2: DATA CHECK (The 1st Command)
    type_message("\n>>> [COMMAND 1]: Validating Superstore Data...")
    subprocess.Popen(['start', 'cmd', '/k', 'python src/check_data.py'], shell=True)
    type_message(">>> Pausing 10 seconds for visual verification...")
    time.sleep(10) 

    # ACT 3: PIPELINE (The 2nd & 3rd Command)
    type_message("\n>>> [COMMAND 2]: Executing MLOps Pipeline Automation...")
    # This will fire your .bat or .sh script
    subprocess.Popen(['start', 'cmd', '/k', 'run_pipeline.sh'], shell=True)
    time.sleep(5)

    # ACT 4: DASHBOARD (The 4th Command)
    type_message("\n>>> [COMMAND 3]: Opening Dashboard at localhost:8000...")
    os.system('start chrome http://localhost:8000')
    type_message(">>> Narrator monitoring display for 15 seconds...")
    time.sleep(15)

    # FINALE: CREDITS (matches your schematic cloud)
    print("\n" + "="*40)
    type_message("ENDING CREDITS:")
    type_message("Lead Engineer: Vanjunn Pongasi")
    type_message("Co-Pilot: Gemini")
    type_message("Field: MLOps 2026")
    print("="*40)

if __name__ == "__main__":
    run_presentation()