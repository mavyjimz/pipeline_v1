import torch
import torch.nn as nn
import pandas as pd
import os

# --- Configuration ---
PROCESSED_DATA = r"D:\MLOps\input_data\processed\cleaned_sales.csv"
MODEL_SAVE_PATH = r"D:\MLOps\models\sales_predictor.pkl"

# 1. Simple Neural Network
class SalesModel(nn.Module):
    def __init__(self):
        super(SalesModel, self).__init__()
        self.layer = nn.Linear(3, 1) 
    def forward(self, x):
        return self.layer(x)

def train():
    if not os.path.exists(PROCESSED_DATA):
        print("❌ Error: No processed data found!")
        return

    df = pd.read_csv(PROCESSED_DATA)
    
    # NEW: Encode categories on the fly if the column is missing
    if 'Category_Encoded' not in df.columns:
        print("💡 Encoding categories automatically...")
        df['Category_Encoded'] = df['Category'].astype('category').cat.codes
        df['Region_Encoded'] = df['Region'].astype('category').cat.codes
        df['Segment_Encoded'] = df['Segment'].astype('category').cat.codes
    
    X = torch.tensor(df[['Category_Encoded', 'Region_Encoded', 'Segment_Encoded']].values, dtype=torch.float32)
    y = torch.tensor(df[['Sales']].values, dtype=torch.float32)
    
    model = SalesModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("🚀 Training started on RX 580...")
    for epoch in range(100):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # THE FIX: Using torch.save instead of pickle
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model, MODEL_SAVE_PATH)
    print(f"✅ SUCCESS: Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()