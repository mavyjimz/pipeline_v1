import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model():
    MLOPS_ROOT = r"D:\MLOps"
    input_file = os.path.join(MLOPS_ROOT, 'input_data', 'processed', 'sales_summary.csv')
    
    if not os.path.exists(input_file):
        print(f"ERROR: No processed data found at {input_file}")
        return

    df = pd.read_csv(input_file)

    # 1. THE BOUNCER: Only allow numbers
    df_numeric = df.select_dtypes(include=[np.number])
    
    # 2. THE JANITOR: Remove any remaining NaNs
    df_clean = df_numeric.dropna().copy()

    if df_clean.empty:
        print("ERROR: No data left after cleaning!")
        return

    # 3. THE NORMALIZATION SHIELD: Force everything between 0 and 1
    # This prevents the 'NaN' weights you were seeing!
    for col in df_clean.columns:
        c_min = df_clean[col].min()
        c_max = df_clean[col].max()
        if c_max > c_min:
            df_clean[col] = (df_clean[col] - c_min) / (c_max - c_min)
        else:
            df_clean[col] = 0.0

    # Define X (features) and y (target)
    X = df_clean.drop(columns=['Sales']).values.astype(np.float32)
    y = df_clean['Sales'].values.reshape(-1, 1).astype(np.float32)

    # 4. THE TRAINING ENGINE
    model = LinearRegression()
    model.fit(X, y)

    # Calculate RMSE
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    # REPORTING
    print(f"\n--- EVALUATION REPORT (LESSON 18) ---")
    print(f"Features Detected: {len(df_clean.columns) - 1}")
    print(f"RMSE: ${rmse:.2f}")
    
    # Check weights to ensure they aren't NaN
    weights = np.nan_to_num(model.coef_[0])
    feature_names = df_clean.drop(columns=['Sales']).columns
    print("\n--- FEATURE WEIGHTS ---")
    for name, weight in zip(feature_names, weights):
        print(f"{name}: {weight:.4f}")

if __name__ == "__main__":
    train_model()