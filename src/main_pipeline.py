import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

# --- PATH HANDSHAKE (Aligned with Lesson #93 YAML) ---
INPUT_PATH = "/app/input_data/processed/cleaned_sales.csv"
OUTPUT_PATH = "/app/shared_output/final_predictions.csv"

def run_pipeline():
    print("🚀 Starting MLOps Pipeline...")

    # 1. Self-Healing: Ensure the 'Loading Dock' exists
    # This fixes the Error you saw earlier by building the folder in Linux
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created missing directory: {output_dir}")

    # 2. Load Data
    try:
        df = pd.read_csv(INPUT_PATH)
        print(f"✅ Data loaded successfully: {len(df)} rows found.")
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find fuel at {INPUT_PATH}. Check Bridge 2!")
        return

    # 3. Hardware Check (RX 580 / CPU Fallback)
    device = torch.device("cpu") # Defaulting to CPU for stability on 9.8k rows
    print(f"💻 Using device: {device}")

    # --- (Logic from Lesson #87: Simple Prediction Simulation) ---
    # We use your existing logic here to keep the 9,800 rows consistent
    df['Predicted_Sales'] = df['Sales'] * 1.05 # Simulating a 5% growth prediction
    
    # 4. Save the Gold
    try:
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"✨ SUCCESS: Gold delivered to {OUTPUT_PATH}")
        print(f"📂 Check your Windows folder: D:\\MLOps\\shared_data")
    except Exception as e:
        print(f"❌ ERROR during save: {e}")

if __name__ == "__main__":
    run_pipeline()