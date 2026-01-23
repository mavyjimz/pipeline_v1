import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Force UTF-8 for the terminal to prevent UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. THE ARCHITECTURE
class SalesPredictor(nn.Module):
    def __init__(self):
        super(SalesPredictor, self).__init__()
        # Line 19/21 Logic
        self.layer = nn.Linear(3, 1) 

    def forward(self, x):
        return self.layer(x)

def train():
    processed_path = r"D:\MLOps\input_data\processed\cleaned_sales.csv"
    
    if not os.path.exists(processed_path):
        print("ERROR: Processed data not found.")
        return

    df = pd.read_csv(processed_path)

    # 2. ENCODING
    df['Category_Encoded'] = df['Category'].astype('category').cat.codes
    df['Region_Encoded'] = df['Region'].astype('category').cat.codes
    df['Segment_Encoded'] = df['Segment'].astype('category').cat.codes

    # 3. THE MATRIX
    features = ['Category_Encoded', 'Region_Encoded', 'Segment_Encoded']
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df['Sales'].values, dtype=torch.float32).view(-1, 1)

    # 4. TRAINING EXECUTION
    model = SalesPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print("STARTING MULTI-FEATURE TRAINING...")
    try:
        # Line 63 Logic
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X) 
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

        # 5. THE EXPORT
        os.makedirs("models", exist_ok=True)
        torch.save(model, "models/sales_predictor.pkl")
        print("SUCCESS: Multi-feature model saved.")
    
    except Exception as e:
        print(f"CRASH DURING TRAINING: {e}")

if __name__ == "__main__":
    train()