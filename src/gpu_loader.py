import torch
import torch_directml
import pandas as pd
import os
import time

def load_data_to_vram():
    # 1. Connect to your AMD RX 580
    device = torch_directml.device()
    print(f"📡 Device Connected: {device}")

    # 2. Find the CSV in the Warehouse
    file_path = "../../input_data/raw/superstore_sales.csv"
    
    print(f"📂 Reading from Warehouse: {file_path}")
    df = pd.read_csv(file_path)
          
    # 3. Prepare the data for the GPU (only using numbers)
    # We take the 100,000 rows from 'Value_A'
    numerical_values = df['Value_A'].values
    
    # 4. THE JUMP: Move from RAM to VRAM
    print("🚀 Pushing 100,000 rows to RX 580 VRAM...")
    
    # Convert to Tensor and send to DirectML device
    gpu_tensor = torch.tensor(numerical_values, dtype=torch.float32).to(device)
    
    print(f"✅ Success! GPU Tensor Shape: {gpu_tensor.shape}")
    print("💎 The data is now live in your 8GB VRAM.")
    
    # We pause for 5 seconds so you have time to see the VRAM monitor flicker!
    time.sleep(5)

if __name__ == "__main__":
    load_data_to_vram()