import torch
import torch.nn as nn
import torch_directml
import pandas as pd
import os

# --- PATH CONFIGURATION (Phase 7 Lesson 32) ---
# Redirected from 'raw' to 'processed' Golden Path
INPUT_PATH = r"D:\MLOps\input_data\processed\cleaned_sales.csv"
MODEL_PATH = r"D:\MLOps\models\sales_model.pth"
OUTPUT_PATH = r"D:\MLOps\projects\pipeline_v1\reports\final_predictions.csv"

def get_data_from_warehouse():
    print("----------------------------------------------------")
    print("PHASE 7 LESSON 32: LOADING FROM GOLDEN PATH")
    print(f"SOURCE: {INPUT_PATH}")
    
    if not os.path.exists(INPUT_PATH):
        print("ERROR: Sanitized data not found. Run sanitizer.py first.")
        return None, None

    df = pd.read_csv(INPUT_PATH)
    
    # Final Safety Check for RX 580 stability
    if df.isnull().sum().sum() > 0:
        print("CRITICAL ERROR: Cleaned data contains nulls. Aborting.")
        return None, None

    input_size = 25
    batch_tensor = torch.zeros(len(df), input_size)
    
    # Industrial Mapping Logic
    region_map = {"West": 5, "East": 6, "South": 7, "Central": 8}
    category_map = {"Furniture": 0, "Technology": 1, "Office Supplies": 2}
    
    for i, row in df.iterrows():
        reg_idx = region_map.get(row['Region'])
        cat_idx = category_map.get(row['Category'])
        if cat_idx is not None: batch_tensor[i, cat_idx] = 1.0
        if reg_idx is not None: batch_tensor[i, reg_idx] = 1.0
            
    print("SUCCESS: Data verified and vectorized.")
    return batch_tensor, df

def run_grand_prediction():
    # Use DirectML for RX 580
    device = torch_directml.device()
    
    # 1. LOAD DATA
    data_tensor, original_df = get_data_from_warehouse()
    if data_tensor is None: return
    
    data_tensor = data_tensor.to(device)

    # 2. LOAD MODEL
    model = nn.Sequential(
        nn.Linear(25, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}. Using untrained weights.")

    # 3. COMPUTE
    print(f"COMPUTE: Generating {len(original_df)} predictions...")
    with torch.no_grad():
        predictions = model(data_tensor)

    # 4. THE HARVEST
    print("PHASE 3: EXPORTING RESULTS")
    
    # Move predictions back to CPU
    preds_cpu = predictions.cpu().numpy().flatten()
    original_df['Predicted_Sales'] = preds_cpu * 100
    
    # Ensure reports folder exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    original_df.to_csv(OUTPUT_PATH, index=False)
    print(f"SUCCESS: Results saved to {OUTPUT_PATH}")
    print("----------------------------------------------------")

if __name__ == "__main__":
    run_grand_prediction()