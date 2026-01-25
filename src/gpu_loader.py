import torch
import torch_directml
import pandas as pd
import os

def load_production_batch():
    print("[START]: Lesson 27 - Smart GPU Batch Loader")
    device = torch_directml.device()
    
    # 1. Path Configuration
    file_path = r"D:\MLOps\input_data\raw\superstore_sales.csv"
    
    # 2. Extract Data
    if not os.path.exists(file_path):
        print(f"[ERROR]: Source CSV missing at {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f"--- Loaded {len(df)} rows from CSV ---")

    # 3. Transform Text to Model Features (Indices from Lesson 26)
    # We create a 25-feature tensor for all 9,800 rows
    input_size = 25
    batch_tensor = torch.zeros(len(df), input_size).to(device)

    # MAP REGIONS (Example mapping based on our Lesson 26 Logic)
    # West = Index 5, East = Index 6, South = Index 7, Central = Index 8
    region_map = {"West": 5, "East": 6, "South": 7, "Central": 8}
    category_map = {"Furniture": 0, "Technology": 1, "Office Supplies": 2}

    print("--- Converting Text to GPU Tensors (ETL Process) ---")
    
    # Efficiently fill the tensor (Vectorized mapping)
    for i, row in df.iterrows():
        reg_idx = region_map.get(row['Region'])
        cat_idx = category_map.get(row['Category'])
        
        if cat_idx is not None: batch_tensor[i, cat_idx] = 1.0
        if reg_idx is not None: batch_tensor[i, reg_idx] = 1.0

    print(f"[SUCCESS]: Pushed {batch_tensor.shape} to RX 580 VRAM.")
    return batch_tensor

if __name__ == "__main__":
    load_production_batch()