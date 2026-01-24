import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch_directml # Critical for AMD RX 580 on Windows

def train_model():
    print("[START]: Lesson 21 AI Training Heartbeat...")
    
    # 1. Device Setup for AMD RX 580
    # Using DirectML instead of CUDA for AMD compatibility
    device = torch_directml.device()
    print(f"--- Hardware Check: Training on {device} (AMD) ---")

    # 2. Load the 25 features engineered in Lesson 19
    processed_path = r"D:\MLOps\input_data\processed\sales_summary.csv"
    df = pd.read_csv(processed_path)
    
    X = df.drop(columns=['Sales']).values
    y = df['Sales'].values.reshape(-1, 1)

    # 3. 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Convert to Tensors and move to AMD GPU
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    # 5. Model Architecture (25 inputs) moved to GPU
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    # 6. Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 7. The Training Heartbeat (100 Epochs)
    print("--- Commencing Training Loop ---")
    for epoch in range(1, 101):
        # Forward Pass
        predictions = model(X_train)
        loss = criterion(predictions, y_train)

        # Backward Pass (The Learning)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Monitor every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/100] | Loss: {loss.item():.4f}")

    # 8. Save the Brain
    model_save_path = r"D:\MLOps\models\sales_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"[SUCCESS]: Model Born and Saved to {model_save_path}")

    return model

if __name__ == "__main__":
    train_model()