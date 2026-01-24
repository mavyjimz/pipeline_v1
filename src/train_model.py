import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
import pandas as pd
import numpy as np
import os

def train():
    print("[START]: Lesson 21 - Training the Unified 25-Feature Brain")
    
    # 1. Hardware Sync
    device = torch_directml.device()
    print(f"--- Training on Hardware: {device} ---")

    # 2. Path to Warehouse
    data_path = r"D:\MLOps\input_data\processed\sales_summary.csv"
    model_save_path = r"D:\MLOps\models\sales_model.pth"

    # 3. Load and Force Numeric Format
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Sales']).values
    y = df['Sales'].values

    # CRITICAL FIX: Force all data to float32 to prevent the 'object_' error
    X = X.astype(np.float32) 
    y = y.astype(np.float32)

    INPUT_SIZE = X.shape[1]
    print(f"--- Sync Check: Detected {INPUT_SIZE} features for training ---")

    # Convert to Tensors for RX 580
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

    # 4. Define 25-Input Architecture
    model = nn.Sequential(
        nn.Linear(INPUT_SIZE, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    # 5. Optimization
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Training Loop (100 Epochs)
    print("--- Training in Progress... ---")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # 7. Save Brain
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[SUCCESS]: Brain saved to {model_save_path}")

if __name__ == "__main__":
    train()