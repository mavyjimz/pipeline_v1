import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

def train_model():
    print("🚀 Lesson 20: Starting AI Training Pipeline...")
    
    # 1. Load the 25 features we engineered in Lesson 19
    df = pd.read_csv("data/processed/superstore_cleaned.csv")
    
    X = df.drop(columns=['Sales']).values
    y = df['Sales'].values.reshape(-1, 1)

    # 2. 80/20 Split (Crucial for Deployment)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Convert to Tensors for RX 580 (CUDA)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    # 4. Simple Neural Network Architecture (25 inputs)
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    print(f"✅ Model Initialized with {X_train.shape[1]} Input Features.")
    return model

if __name__ == "__main__":
    train_model()