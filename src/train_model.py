import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
import pandas as pd
import numpy as np
import os

def train():
    print("[START]: Lesson 23 - Training with History Tracking")
    
    device = torch_directml.device()
    data_path = r"D:\MLOps\input_data\processed\sales_summary.csv"
    model_save_path = r"D:\MLOps\models\sales_model.pth"
    history_path = r"D:\MLOps\reports\loss_history.csv" # New path for Lesson 23

    # Load and Force Numeric
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Sales']).values.astype(np.float32)
    y = df['Sales'].values.astype(np.float32)

    X_tensor = torch.tensor(X).to(device)
    y_tensor = torch.tensor(y).view(-1, 1).to(device)

    model = nn.Sequential(
        nn.Linear(X.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # --- LESSON 23: HISTORY TRACKER ---
    loss_history = [] 

    print("--- Training & Recording History on RX 580 ---")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Save the loss value to our list
        loss_history.append(loss.item())
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # Save Model
    torch.save(model.state_dict(), model_save_path)
    
    # --- LESSON 23: SAVE HISTORY TO CSV ---
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    pd.DataFrame({"epoch": range(1, 101), "loss": loss_history}).to_csv(history_path, index=False)
    
    print(f"[SUCCESS]: Brain and Loss History saved to Warehouse.")

if __name__ == "__main__":
    train()