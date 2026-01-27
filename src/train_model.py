import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
import pandas as pd
import os

# --- PATH CONFIGURATION ---
INPUT_PATH = r"D:\MLOps\input_data\processed\cleaned_sales.csv"
MODEL_SAVE_PATH = r"models\sales_model.pth"
HISTORY_PATH = r"D:\MLOps\projects\pipeline_v1\reports\training_history.csv"

def train_factory():
    print("----------------------------------------------------")
    print("PHASE 7 LESSON 34: OPTIMIZED TRAINING (TEMPORAL)")
    print("----------------------------------------------------")

    # 1. HARDWARE HANDSHAKE
    device = torch_directml.device()
    print(f"HARDWARE: Training on RX 580 via {device}")

    # 2. LOAD GOLDEN PATH DATA
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Golden Path file not found at {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)
    
    # 3. VECTORIZATION (27 Dimensions)
    input_size = 27
    X = torch.zeros(len(df), input_size)
    
    region_map = {"West": 5, "East": 6, "South": 7, "Central": 8}
    category_map = {"Furniture": 0, "Technology": 1, "Office Supplies": 2}
    
    for i, row in df.iterrows():
        # Category & Region One-Hot
        cat_idx = category_map.get(row['Category'])
        reg_idx = region_map.get(row['Region'])
        if cat_idx is not None: X[i, cat_idx] = 1.0
        if reg_idx is not None: X[i, reg_idx] = 1.0
        
        # Temporal Features (Lesson 33)
        X[i, 25] = float(row['Order_Month']) / 12.0
        X[i, 26] = float(row['Order_DayOfWeek']) / 6.0

    # Target: Sales (Scaled for stability)
    y = torch.tensor(df['Sales'].values, dtype=torch.float32).view(-1, 1) / 100.0
    
    X, y = X.to(device), y.to(device)

    # 4. ARCHITECTURE (The 27-64-1 Brain)
    model = nn.Sequential(
        nn.Linear(27, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    # 5. OPTIMIZATION SETTINGS
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = []

    # 6. TRAINING LOOP
    print("ACTION: Starting 100 Epochs of Gradient Descent...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        history.append({"epoch": epoch + 1, "loss": loss.item()})
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # 7. SAVE OUTPUTS
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Save History for visualization later
    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)
    
    print(f"SUCCESS: Model saved to {MODEL_SAVE_PATH}")
    print(f"SUCCESS: History logged to {HISTORY_PATH}")
    print("----------------------------------------------------")

if __name__ == "__main__":
    train_factory()