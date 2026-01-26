import torch
import torch.nn as nn
import torch_directml
import pandas as pd
import os

def get_data_from_warehouse():
    file_path = r"D:\MLOps\input_data\raw\superstore_sales.csv"
    df = pd.read_csv(file_path)
    # ... (Your existing loading logic) ...
    input_size = 25
    batch_tensor = torch.zeros(len(df), input_size)
    region_map = {"West": 5, "East": 6, "South": 7, "Central": 8}
    category_map = {"Furniture": 0, "Technology": 1, "Office Supplies": 2}
    for i, row in df.iterrows():
        reg_idx = region_map.get(row['Region'])
        cat_idx = category_map.get(row['Category'])
        if cat_idx is not None: batch_tensor[i, cat_idx] = 1.0
        if reg_idx is not None: batch_tensor[i, reg_idx] = 1.0
    return batch_tensor, df # We return the DF too so we can save it later!

def run_grand_prediction():
    device = torch_directml.device()
    
    # 1. LOAD DATA
    data_tensor, original_df = get_data_from_warehouse()
    data_tensor = data_tensor.to(device)

    # 2. LOAD MODEL (Your existing Sequential logic)
    model = nn.Sequential(nn.Linear(25, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    model.load_state_dict(torch.load(r"D:\MLOps\models\sales_model.pth", map_location=device))
    model.eval()

    # 3. COMPUTE
    print("[COMPUTE]: Generating 9,800 predictions...")
    with torch.no_grad():
        predictions = model(data_tensor)

    # 4. THE HARVEST (New Lesson 30 Logic)
    print("--- [PHASE 3: EXPORTING RESULTS] ---")
    
    # Move predictions from GPU back to CPU and convert to a list
    # We multiply by 100 to reverse any scaling if needed
    original_df['Predicted_Sales'] = predictions.cpu().numpy().flatten() * 100
    
    output_path = r"D:\MLOps\projects\pipeline_v1\reports\final_predictions.csv"
    
    # Ensure the reports folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    original_df.to_csv(output_path, index=False)
    print(f"✅ SUCCESS: Results saved to {output_path}")

if __name__ == "__main__":
    run_grand_prediction()