import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# --- 1. SETTINGS & ABSOLUTE PATH LAW ---
MLOPS_ROOT = r"D:\MLOps"
DATA_PATH = os.path.join(MLOPS_ROOT, "input_data", "sales_data.csv")
MODEL_DIR = os.path.join(MLOPS_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_pipeline():
    # --- 2. DATA LOADING & ENCODING ---
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # One-Hot Encoding (Categorical to Numerical)
    df = pd.get_dummies(df, columns=['Category', 'Region', 'Segment'], dtype=float)
    
    # Separate Features (X) and Target (y)
    X = df.drop(columns=['Sales', 'Order Date', 'Customer Name'], errors='ignore')
    y = df['Sales']

    # Convert to Tensors (float32 for RX 580 compatibility)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    # --- 3. THE NEURAL NETWORK ---
    input_dim = X_tensor.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # --- 4. TRAINING LOOP ---
    print(f"Starting Training on {input_dim} features...")
    for epoch in range(100):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # --- 5. LESSON 16: EVALUATION METRICS ---
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor)
        # Convert back to numpy for Sklearn metrics
        preds_np = predictions.numpy()
        actuals_np = y_tensor.numpy()
        
        mse = mean_squared_error(actuals_np, preds_np)
        rmse = np.sqrt(mse)
        
    print("\n--- 📊 EVALUATION REPORT ---")
    print(f"Final MSE: {mse:.4f}")
    print(f"RMSE (Avg Error): ${rmse:.2f}")

    # --- 6. SAVE WEIGHTS ---
    save_path = os.path.join(MODEL_DIR, "sales_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to: {save_path}")

if __name__ == "__main__":
    train_pipeline()