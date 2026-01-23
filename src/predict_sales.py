import torch
import pandas as pd
import os
from train_model import SalesPredictor # Importing the 'Brain' structure

def predict():
    # --- THE LAW: ABSOLUTE GPS PATHS ---
    MLOPS_ROOT = r"D:\MLOps"
    MODEL_PATH = os.path.join(MLOPS_ROOT, "models", "sales_model.pth")
    DATA_PATH = os.path.join(MLOPS_ROOT, "input_data", "processed", "sales_summary.csv")
    
    # 1. CHECK IF EVIDENCE EXISTS
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: No processed data at {DATA_PATH}")
        return

    # 2. PREPARE THE DATA (Lesson 15 One-Hot Encoding)
    df = pd.read_csv(DATA_PATH)
    required_cols = ['Category', 'Region', 'Segment']
    
    # We apply the same transformation used in training
    df_encoded = pd.get_dummies(df, columns=required_cols)
    X_df = df_encoded.select_dtypes(include=['number', 'bool']).copy()
    
    if 'Sales' in X_df.columns:
        X_df = X_df.drop(columns=['Sales'])

    # 3. INITIALIZE AND LOAD THE BORN MODEL
    input_dim = X_df.shape[1]
    model = SalesPredictor(input_dim)
    
    # Load weights (using 2026 security standards)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval() # Tell the model: "We are predicting, not learning"

    # 4. MAKE A PREDICTION (Using the first row of your clean data)
    sample_input = torch.tensor(X_df.iloc[:1].values.astype(float), dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(sample_input)
    
    print(f"\n🔮 PREDICTION SYSTEM ONLINE")
    print(f"Input Features: {list(X_df.columns)}")
    print(f"Predicted Sales Result: ${prediction.item():.2f}")

if __name__ == "__main__":
    predict()