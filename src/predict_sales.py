import torch
import torch.nn as nn
import torch_directml
import os

def run_prophecy():
    print("[START]: Lesson 22 - The Sales Prophet Engine")
    
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

    # 5. Generate a Prediction
    # We create a dummy input of 25 features to test the math
    # In a real run, this would be a row from your sales_summary.csv
    sample_input = torch.randn(1, INPUT_SIZE).to(device)

    print("--- Executing Prediction on RX 580 ---")
    with torch.no_grad():
        prediction = model(sample_input)
        
    print("-----------------------------------------")
    print(f"PROPHET RESULT (Sales): {prediction.item():.4f}")
    print("-----------------------------------------")
    print("[DONE]: Lesson 22 Complete.")

if __name__ == "__main__":
    run_prophecy()