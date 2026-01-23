import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import os

# 1. Setup Paths
DATA_PATH = r"D:\MLOps\input_data\processed\sales_summary.csv"
MODEL_SAVE_PATH = r"D:\MLOps\models\sales_model.pth"

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}")
        return

    # 2. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # SPIDER-SENSE SHIELD: Only use numeric data for the RX 580
    df_numeric = df.select_dtypes(include=[np.number])
    
    X = df_numeric.drop(columns=['Sales']).values.astype(np.float32)
    y = df_numeric['Sales'].values.reshape(-1, 1).astype(np.float32)
    
    feature_names = df_numeric.drop(columns=['Sales']).columns.tolist()

    # 3. Model Definition (Linear Regression in PyTorch)
    input_dim = X.shape[1]
    model = nn.Linear(input_dim, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 4. Training Loop
    print(f"--- STARTING LESSON 17 TRAINING ({len(feature_names)} Features) ---")
    for epoch in range(100):
        inputs = torch.from_numpy(X)
        targets = torch.from_numpy(y)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

    # 5. Evaluation Metrics
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X)).numpy()
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)

    print("\n--- EVALUATION REPORT ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

    # 6. FEATURE IMPORTANCE (The Lesson 17 Upgrade)
    print("\n--- FEATURE IMPORTANCE (Weights) ---")
    weights = model.weight.data.numpy().flatten()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': weights
    }).sort_values(by='Weight', ascending=False)
    
    print(importance_df.to_string(index=False))

    # 7. Save Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nSUCCESS: Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()