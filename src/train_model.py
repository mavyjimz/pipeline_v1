import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os

# 1. THE BRAIN
class SalesPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SalesPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train():
    # --- THE LAW: ABSOLUTE GPS PATHS ---
    MLOPS_ROOT = r"D:\MLOps"
    PROCESSED_DATA_PATH = os.path.join(MLOPS_ROOT, "input_data", "processed", "sales_summary.csv")
    MODEL_WAREHOUSE = os.path.join(MLOPS_ROOT, "models")
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"ERROR: No processed data at {PROCESSED_DATA_PATH}. Run clean_data.py first!")
        return

    # 2. LOAD DATA
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 3. LESSON 15: ONE-HOT ENCODING
    required_cols = ['Category', 'Region', 'Segment']
    df_encoded = pd.get_dummies(df, columns=required_cols)
    
    # 4. TITANIUM FILTER (Protecting RX 580)
    X_df = df_encoded.select_dtypes(include=['number', 'bool']).copy()
    if 'Sales' in X_df.columns:
        X_df = X_df.drop(columns=['Sales'])

    # Convert to Float32 Tensors
    X = torch.tensor(X_df.values.astype(float), dtype=torch.float32)
    y = torch.tensor(df['Sales'].values.astype(float), dtype=torch.float32).view(-1, 1)

    # 5. INITIALIZE & TRAIN
    input_dim = X.shape[1]
    model = SalesPredictor(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print(f"\nSTARTING TRAINING: Inputs = {input_dim} features")
    for epoch in range(100):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # 6. BANISHING THE GHOST (Absolute Path Saving)
    os.makedirs(MODEL_WAREHOUSE, exist_ok=True)
    save_path = os.path.join(MODEL_WAREHOUSE, "sales_model.pth")
    torch.save(model.state_dict(), save_path)
    
    print(f"\nMISSION ACCOMPLISHED: Model saved to {save_path}")

if __name__ == "__main__":
    train()