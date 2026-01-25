import torch
import torch.nn as nn
import torch_directml
import os

def run_prophet():
    print("[START]: Lesson 25 - The Sales Decoder Engine")

    # 1. Hardware Initialization (RX 580 Optimization)
    device = torch_directml.device()
    print(f"--- Hardware Active: {device} ---")

    # 2. Configuration (The 25-Feature Contract)
    INPUT_SIZE = 25 
    model_path = r"D:\MLOps\models\sales_model.pth"

    # 3. Build the Architecture (The Mirror)
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    # 4. Load the Brain from the Warehouse
    if os.path.exists(model_path):
        # weights_only=False is required for AMD/DirectML state loads
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        print(f"[SUCCESS]: 25-Feature Brain loaded successfully.")
    else:
        print(f"[ERROR]: Brain file not found at {model_path}!")
        return

    # 5. Targeted Scenario (Furniture @ West Region)
    sample_input = torch.zeros(1, INPUT_SIZE).to(device)
    sample_input[0, 0] = 1.0  # Index 0: Furniture
    sample_input[0, 5] = 1.0  # Index 5: West Region

    print("--- Executing Decoded Prediction on RX 580 ---")
    with torch.no_grad():
        prediction = model(sample_input)
        raw_value = prediction.item()
        
        # LESSON 25: THE DECODER
        # We assume a scaling factor of 100 from our training phase
        decoded_sales = raw_value * 100

    print("---------------------------------------")
    print(f"RAW AI SIGNAL    : {raw_value:.4f}")
    print(f"DECODED PREDICTION: ${decoded_sales:,.2f} USD")
    print("---------------------------------------")
    print("[DONE]: Lesson 25 Complete.")

if __name__ == "__main__":
    run_prophet()