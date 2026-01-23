import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os

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
    processed_path = r"D:\MLOps\input_data\processed\sales_summary.csv"
    
    if not os.path.exists(processed_path):
        print(f"ERROR: No processed data at {processed_path}. Run clean_data.py first!")
        return

    df = pd.read_csv(processed_path)

    # LESSON 15: One-Hot Encoding the required columns
    required_cols = ['Category', 'Region', 'Segment']
    df_encoded = pd.get_dummies(df, columns=required_cols)
    
    # Filter for numbers and bools only (Protect RX 580)
    X_df = df_encoded.select_dtypes(include=['number', 'bool']).copy()
    if 'Sales' in X_df.columns:
        X_df = X_df.drop(columns=['Sales'])

    X = torch.tensor(X_df.values.astype(float), dtype=torch.float32)
    y = torch.tensor(df['Sales'].values.astype(float), dtype=torch.float32).view(-1, 1)

    model = SalesPredictor(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print(f"\nSTARTING TRAINING: Inputs = {X.shape[1]} features")
    for epoch in range(100):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/sales_model.pth")
    print("\nMISSION ACCOMPLISHED: Model is Born!")

if __name__ == "__main__":
    train()