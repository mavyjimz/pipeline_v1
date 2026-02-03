import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def run_training():
    # 1. Load Data
    input_file = 'shared_output/cleaned_sales_data.csv'
    if not os.path.exists(input_file):
        print("Error: Cleaned data not found. Please ensure Phase 2 was successful.")
        return

    df = pd.read_csv(input_file)
    
    # 2. Define Features (X) and Target (y)
    try:
        y = df['Sales']
        X = df[['Quantity', 'Discount', 'Profit']] 
    except KeyError as e:
        print(f"Error: Missing column in dataset: {e}")
        return

    # 3. Data Partitioning (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Data Split Successful: {len(X_train)} training rows, {len(X_test)} testing rows.")

    # 4. Initialize and Train Model on CPU
    print("Initializing Random Forest training on CPU...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Model Serialization
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, 'sales_model.pkl')
    joblib.dump(model, model_path)
    print(f"Success: Model saved to: {model_path}")

if __name__ == "__main__":
    run_training()