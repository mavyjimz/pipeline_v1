import torch
import torch.nn as nn
import torch_directml
import pandas as pd
import os

# --- PATH CONFIGURATION ---
INPUT_PATH = r"D:\MLOps\input_data\processed\cleaned_sales.csv"
MODEL_PATH = r"models\sales_model.pth"
OUTPUT_PATH = r"D:\MLOps\projects\pipeline_v1\reports\final_predictions.csv"

def get_data_from_warehouse():
    print("----------------------------------------------------")
    print("PHASE 7 LESSON 33: TEMPORAL INTEGRATION")
    print(f"SOURCE: {INPUT_PATH}")
    
    if not os.path.exists(INPUT_PATH):
        print("ERROR: Sanitized data not found.")
        return None, None

    df = pd.read_csv(INPUT_PATH)
    
    # NEW INPUT SIZE: 25 (Old) + 1 (Month) + 1 (DayOfWeek) = 27
    input_size = 27
    batch_tensor = torch.zeros(len(df), input_size)
    
    region_map = {"West": 5, "East": 6, "South": 7, "Central": 8}
    category_map = {"Furniture": 0, "Technology": 1, "Office Supplies": 2}
    
    for i, row in df.iterrows():
        # A. Original Logic
        reg_idx = region_map.get(row['Region'])
        cat_idx = category_map.get(row['Category'])
        if cat_idx is not None: batch_tensor[i, cat_idx] = 1.0
        if reg_idx is not None: batch_tensor[i, reg_idx] = 1.0
        
        # B. TEMPORAL ATTACK (New Lesson 33 Features)
        # Monthly signal (normalized 0-1 for RX 580 stability)
        batch_tensor[i, 25] = float(row['Order_Month']) / 12.0
        # Day of Week signal (normalized 0-1)
        batch_tensor[i, 26] = float(row['Order_DayOfWeek']) / 6.0
            
    print("SUCCESS: Temporal features vectorized (27-dim).")
    return batch_tensor, df

def run_grand_prediction():
    device = torch_directml.device()
    
    data_tensor, original_df = get_data_from_warehouse()
    if data_tensor is None: return
    data_tensor = data_tensor.to(device)

    # UPDATED MODEL ARCHITECTURE (Input 27)
    model = nn.Sequential(
        nn.Linear(27, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    
    # Note: Loading old weights might cause a size mismatch warning. 
    # That is expected when we grow the brain!
    if os.path.exists(MODEL_PATH):
        print("NOTE: Attempting to load existing weights...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
            print("Loaded partial weights (Transfer Learning active).")
        except:
            print("Size mismatch: Model brain has grown. Using fresh initialization.")

    model.eval()
    print(f"COMPUTE: Processing {len(original_df)} rows with Temporal Awareness...")
    with torch.no_grad():
        preds = model(data_tensor)

    preds_cpu = preds.cpu().numpy().flatten()
    original_df['Predicted_Sales'] = preds_cpu * 100
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    original_df.to_csv(OUTPUT_PATH, index=False)
    print(f"SUCCESS: Results saved to {OUTPUT_PATH}")
    print("----------------------------------------------------")

if __name__ == "__main__":
    run_grand_prediction()