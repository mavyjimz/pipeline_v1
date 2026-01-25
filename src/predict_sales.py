import torch
import torch.nn as nn
import torch_directml
import os

def run_prophecy():
    print("[START]: The Sales Prophet Engine")
    
    # 1. Hardware Initialization
    device = torch_directml.device()
    print(f"--- Hardware Active: {device} ---")

    # 2. Configuration (The Contract)
    INPUT_SIZE = 25  # This MUST match the Brain we trained
    model_path = r"D:\MLOps\models\sales_model.pth"

    # 3. Build the Architecture (The Mirror)
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    # 4. Load the Brain from the Warehouse
    if os.path.exists(model_path):
        # We use weights_only=False to ensure the AMD state loads correctly
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        print(f"[SUCCESS]: 25-Feature Brain loaded successfully.")
    else:
        print(f"[ERROR]: Brain file not found at {model_path}!")
        return

    # 5. Generate a Real Prediction (Lesson 24 Upgrade)
    # Instead of random noise, we create a zeroed tensor
    sample_input = torch.zeros(1, INPUT_SIZE).to(device)

    # We "flip the switches" for a specific scenario:
    # Let's assume: Index 0 = Furniture, Index 5 = West Region
    sample_input[0, 0] = 1.0 
    sample_input[0, 5] = 1.0 

    print("--- Executing Targeted Scenario Prediction on RX 580 ---")
    with torch.no_grad():
        prediction = model(sample_input)
        
    print("-----------------------------------------")
    print(f"PROPHET RESULT (Sales): {prediction.item():.4f}")
    print("-----------------------------------------")
    print("[DONE]: Complete.")

if __name__ == "__main__":
    run_prophecy()