import torch
import torch_directml
import pandas as pd
import time

def start_race():
    # 1. Setup Devices
    dml = torch_directml.device()
    cpu = torch.device("cpu")
    
    # 2. Get Data from Warehouse (D: Drive)
    file_path = r'D:\MLOps\input_data\raw\test_data.csv'
    
    try:
        df = pd.read_csv(file_path)
        # Using 'Value_A' to match your intake_valve.py output
        data_tensor = torch.tensor(df['Value_A'].values, dtype=torch.float32)
    except KeyError:
        print("❌ Error: Column 'Value_A' not found. Check your CSV header!")
        return

    print(f"🏁 Starting Race: 100,000 rows x 5,000 iterations!")
    print("-" * 40)

    # --- ROUND 1: CPU ---
    print("🐌 CPU is starting...")
    start_cpu = time.time()
    cpu_data = data_tensor.to(cpu)
    for _ in range(5000):
        # Heavy math: squaring the numbers
        cpu_result = torch.pow(cpu_data, 2)
    end_cpu = time.time()
    print(f"✅ CPU Total Time: {end_cpu - start_cpu:.4f} seconds")

    # --- ROUND 2: GPU (RX 580) ---
    print("\n🚀 RX 580 is starting...")
    start_gpu = time.time()
    gpu_data = data_tensor.to(dml) # Move data to VRAM
    for _ in range(5000):
        # Same heavy math on the GPU
        gpu_result = torch.pow(gpu_data, 2)
    end_gpu = time.time()
    print(f"✅ GPU Total Time: {end_gpu - start_gpu:.4f} seconds")
    print("-" * 40)

if __name__ == "__main__":
    start_race()