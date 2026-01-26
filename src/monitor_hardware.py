import GPUtil
import time
import torch_directml

def monitor_rx580():
    print("--- [GPU TELEMETRY ACTIVE] ---")
    device = torch_directml.device()
    
    try:
        while True:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"GPU: {gpu.name} | Temp: {gpu.temperature}°C | "
                      f"VRAM: {gpu.memoryUsed}/{gpu.memoryTotal}MB | "
                      f"Load: {gpu.load*100}%")
            
            time.sleep(1) # Check every second
    except KeyboardInterrupt:
        print("\n[STOPPED]: Monitoring finished.")

if __name__ == "__main__":
    monitor_rx580()