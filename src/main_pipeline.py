import torch
import torch.nn as nn
import torch_directml
import pandas as pd
import os

# --- PART 1: THE SMART LOADER (Lesson 27) ---
def get_data_from_warehouse():
    file_path = r"D:\MLOps\input_data\raw\superstore_sales.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find CSV at {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"[LOADER]: Successfully read {len(df)} rows from CSV.")
    
    # Create the empty 25-feature tensor
    input_size = 25
    batch_tensor = torch.zeros(len(df), input_size)

    # Mapping Logic (Matching our Lesson 26 logic)
    region_map = {"West": 5, "East": 6, "South": 7, "Central": 8}
    category_map = {"Furniture": 0, "Technology": 1, "Office Supplies": 2}

    for i, row in df.iterrows():
        reg_idx = region_map.get(row['Region'])
        cat_idx = category_map.get(row['Category'])
        if cat_idx is not None: batch_tensor[i, cat_idx] = 1.0
        if reg_idx is not None: batch_tensor[i, reg_idx] = 1.0
        
    return batch_tensor

# --- PART 2: THE EXECUTION ENGINE (Lesson 28) ---
def run_grand_prediction():
    print("="*50)
    print("🚀 PIPELINE v1.2: FULL DATASET INFERENCE")
    print("="*50)

    # 1. Hardware Setup
    device = torch_directml.device()
    print(f"[DEVICE]: Utilizing {device} (RX 580)")

    # 2. Load Data
    data_tensor = get_data_from_warehouse().to(device)

    # 3. Initialize Model Architecture
    model = nn.Sequential(
        nn.Linear(25, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    # 4. Load the Brain Weights
    model_path = r"D:\MLOps\models\sales_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"[MODEL]: Weights loaded successfully.")
    else:
        print("[ERROR]: sales_model.pth not found on Drive D!")
        return

    # 5. THE STRESS TEST (Lesson 29.5)
    print(f"[STRESS]: Running 9,800 rows x 1,000 loops...")
    
    with torch.no_grad():
        # This loop keeps the GPU busy so we can see the telemetry move!
        for i in range(1000):
            predictions = model(data_tensor)
            if i % 200 == 0:
                print(f"   > Loop {i}/1000 complete...")

    print("="*50)
    print(f"✅ SUCCESS: Stress test complete!")
    print(f"Final Prediction: ${predictions[0].item() * 100:,.2f}")
    print("="*50)

if __name__ == "__main__":
    run_grand_prediction()