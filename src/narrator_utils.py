import sys
import time
import winsound
import random

def type_message(message, delay=0.06):
    """Standard narration typing with sound for the story."""
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        # High-pitched click sound
        winsound.Beep(1200, 10) 
        time.sleep(delay)
    print() 

def fast_scan(lines_count=15):
    """Lightning-fast system logs for that 'Hacker' visual noise."""
    log_messages = [
        "INITIALIZING KERNEL", "MAPPING VIRTUAL MEMORY", "LOADING DOCKER_ENGINE",
        "SYNCING VENV_LIBRARIES", "CHECKING CPU_AFFINITY", "ESTABLISHING LOCALHOST:8000",
        "MOUNTING D:/DRIVE", "VALIDATING SCHEMAS", "DECRYPTING DATA_LAYERS"
    ]
    
    for _ in range(lines_count):
        hex_code = f"0x{random.randint(1000, 9999)}A{random.randint(10, 99)}"
        msg = random.choice(log_messages)
        # Fast green-style log output
        sys.stdout.write(f">>> [{hex_code}] {msg} ... [OK]\n")
        sys.stdout.flush()
        time.sleep(0.03) # High speed
    print(">>> ALL SYSTEMS NOMINAL. HANDING OVER TO NARRATOR...\n")
    time.sleep(1)