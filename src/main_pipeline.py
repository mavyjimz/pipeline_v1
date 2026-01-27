import torch
import torch.nn as nn
import pandas as pd
import os
import time

# --- HARDWARE FALLBACK LOGIC ---
try:
    import torch_directml
    device = torch_directml.device()
    print("SUCCESS: Hardware Acceleration (DirectML) active.")
except (ImportError, OSError):
    device = torch.device("cpu")
    print("NOTE: DirectML not found. Falling back to CPU Mode.")

# --- PATH CONFIGURATION (Internal Container Paths) ---
INPUT_PATH = "/app/input_data/processed/cleaned_sales.csv"
MODEL_PATH = "/app/models/sales_model.pth"
OUTPUT_PATH = "/app/reports/final_predictions.csv"
LOG_PATH = "/app/logs/pipeline_log.txt"  # New for Lesson #42

def get_data_from_warehouse():
    print("-------------------------------------------")
    print("PHASE 8 LESSON #42: CONTAINERIZED EXECUTION")
    print(f"SOURCE: {INPUT_PATH}")

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: File not found at {INPUT_PATH}")
        return None, None

    df = pd.read_csv(INPUT_PATH)
    input_size = 27
    batch_tensor = torch.zeros(len(df), input_size)

    region_map = {"West": 5, "East": 6, "South": 7, "Central": 8}
    category_map = {"Furniture": 0, "Technology": 1, "Office Supplies": 2}

    for i, row in df.iterrows():
        reg_idx = region_map.get(row['Region'])
        cat_idx = category_map.get(row['Category'])
        if cat_idx is not None: batch_tensor[i, cat_idx] = 1.0
        if reg_idx is not None: batch_tensor[i, reg_idx] = 1.0
        
        batch_tensor[i, 25] = float(row['Order_Month']) / 12.0
        batch_tensor[i, 26] = float(row['Order_DayOfWeek']) / 6.0

    print("SUCCESS: Temporal features vectorized (27-dim).")
    return batch_tensor, df

def run_grand_prediction():
    data_tensor, original_df = get_data_from_warehouse()
    if data_tensor is None: return

    data_tensor = data_tensor.to(device)

    model = nn.Sequential(
        nn.Linear(27, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    if os.path.exists(MODEL_PATH):
        print("NOTE: Attempting to load existing weights...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, strict=False))
            print("Loaded partial weights (Transfer Learning active).")
        except Exception as e:
            print(f"Fresh initialization: {e}")

    model.eval()
    print(f"COMPUTE: Processing {len(original_df)} rows on {device}...")

    with torch.no_grad():
        preds = model(data_tensor)

    preds_cpu = preds.cpu().numpy().flatten()
    original_df['Predicted_Sales'] = preds_cpu * 100

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    original_df.to_csv(OUTPUT_PATH, index=False)
    print(f"SUCCESS: Results saved to {OUTPUT_PATH}")

    # --- LESSON #42: PERSISTENT LOGGING ---
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"[{pd.Timestamp.now()}] Run Successful. Rows: {len(original_df)} | Device: {device}\n")
    print(f"SUCCESS: Audit trail updated in {LOG_PATH}")
    print("-------------------------------------------")

if __name__ == "__main__":
    run_grand_prediction()
    
    # Lesson #42: Stay Alive Buffer (300 seconds)
    print("Container staying active for 5 minutes for health monitoring...")
    time.sleep(300)