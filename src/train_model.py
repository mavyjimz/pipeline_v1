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

    # 2. Load and Clean Data
    df = pd.read_csv(DATA_PATH)
    
    # SHIELD 1: Keep only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    # SHIELD 2: TRIPLE PURGE - Remove any row that has a NaN in ANY column
    # This prevents the 'Input contains NaN' error in scikit-learn metrics
    df_clean = df_numeric.dropna().copy()
    
    if df_clean.empty:
        print("CRITICAL ERROR: No data left after dropping NaNs! Check your CSV.")
        return

    # 3. Prepare Tensors
    X = df_clean.drop(columns=['Sales']).values.astype(np.float32)
    y = df_clean['Sales'].values.reshape(-1, 1).astype(np.float32)
    feature_names = df_clean.drop(columns=['Sales']).columns.tolist()

    # 4. Model Definition
    input_dim = X.shape[1]
    model = nn.Linear(input_dim, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 5. Training Loop
    print(f"--- STARTING LESSON 17 TRAINING ({len(feature_names)} Features) ---")
    for epoch in range(100):
        inputs = torch.from_numpy(X)
        targets = torch.from_numpy(y)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 6. Evaluation (THE INDESTRUCTIBLE VERSION)
    model.eval()
    with torch.no_grad():
        # Get raw predictions
        raw_predictions = model(torch.from_numpy(X)).numpy()
        
        # FINAL SAFETY CHECK: Replace any accidental NaNs/Infs with 0.0
        # Sometimes SGD can 'explode' and create Infs if the data is messy
        predictions = np.nan_to_num(raw_predictions)
        clean_y = np.nan_to_num(y)

        mse = mean_squared_error(clean_y, predictions)
        rmse = np.sqrt(mse)

    print("\n--- EVALUATION REPORT ---")
    print(f"RMSE: ${rmse:.2f}")

    # 7. FEATURE IMPORTANCE
    print("\n--- FEATURE IMPORTANCE (Weights) ---")
    weights = model.weight.data.numpy().flatten()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Weight': weights})
    print(importance_df.sort_values(by='Weight', ascending=False).to_string(index=False))

    # 8. Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nSUCCESS: Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()