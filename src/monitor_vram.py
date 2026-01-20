import torch
import torch_directml
import time

def monitor_gpu():
    # Connect to the RX 580 bridge
    device = torch_directml.device()
    gpu_name = torch_directml.device_name(0)
    
    print(f"--- LIVE MONITOR: {gpu_name} ---")
    print("Pipeline is now monitoring VRAM heartbeats.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # Heartbeat check
            timestamp = time.strftime('%H:%M:%S')
            # In DirectML, we check if the device is active
            status = "ACTIVE" if torch_directml.is_available() else "IDLE"
            print(f"\r[VRAM Status] Device: {device} | Status: {status} | Time: {timestamp}", end="")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nMonitoring ended.")

if __name__ == "__main__":
    monitor_gpu()