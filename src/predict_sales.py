import torch
import torch.nn as nn
import torch_directml
import os

def run_prophet():
    print("[START]: Lesson 26 - The Batch Prophet Engine")

    # 1. Hardware Initialization (RX 580 Optimization)
    device = torch_directml.device()
    print(f"--- Hardware Active: {device} ---")

    # 2. Configuration
    INPUT_SIZE = 25
    model_path = r"D:\MLOps\models\sales_model.pth"

    # 3. Build Architecture (The Mirror)
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    # 4. Load the Brain
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        print(f"[SUCCESS]: 25-Feature Brain loaded for Batching.")
    else:
        print(f"[ERROR]: Brain file not found at {model_path}!")
        return

    # 5. BATCH SCENARIOS (Lesson 26 Upgrade)
    # We create 3 scenarios at once: Furniture@West, Tech@East, Office@South
    batch_size = 3
    batch_input = torch.zeros(batch_size, INPUT_SIZE).to(device)

    # Scenario 0: Furniture (Index 0) at West (Index 5)
    batch_input[0, 0] = 1.0 
    batch_input[0, 5] = 1.0

    # Scenario 1: Technology (Index 1) at East (Index 6)
    batch_input[1, 1] = 1.0
    batch_input[1, 6] = 1.0

    # Scenario 2: Office Supplies (Index 2) at South (Index 7)
    batch_input[2, 2] = 1.0
    batch_input[2, 7] = 1.0

    print(f"--- Executing Batch Prediction ({batch_size} rows) on RX 580 ---")
    
    
    with torch.no_grad():
        # The RX 580 processes all 3 rows in one single mathematical operation
        predictions = model(batch_input)
        
    print("-" * 40)
    regions = ["West", "East", "South"]
    categories = ["Furniture", "Technology", "Office Supplies"]

    for i in range(batch_size):
        raw_value = predictions[i].item()
        decoded_sales = raw_value * 100
        print(f"ROW {i} [{categories[i]} @ {regions[i]}]: ${decoded_sales:,.2f} USD")
    
    print("-" * 40)
    print("[DONE]: Lesson 26 Complete.")

if __name__ == "__main__":
    run_prophet()