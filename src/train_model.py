import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Paths
INPUT_FILE = r"D:\MLOps\input_data\processed\cleaned_sales.csv"
MODEL_PATH = r"D:\MLOps\models\sales_model.pkl"

def train():
    print("🧠 AI TRAINING STATION: Learning from data...")
    
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Prepare Data (Group by Category)
    # We convert categories into numbers (0, 1, 2) so the AI can understand them
    df['Category_Code'] = df['Category'].astype('category').cat.codes
    
    X = df[['Category_Code']] # Inputs
    y = df['Sales']           # What we want to predict
    
    # 3. Train the Brain (Linear Regression)
    model = LinearRegression()
    model.fit(X, y)
    
    # 4. Save the "Frozen Brain"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    print(f"✅ Training Complete! Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()