import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def run_training():
    input_file = 'input_data/processed/sales_summary.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Cleaned data not found at {input_file}.")
        return

    df = pd.read_csv(input_file)

    # Separate Target (Sales) from Features
    if 'Sales' not in df.columns:
        print("Error: 'Sales' column missing from processed data.")
        return

    y = df['Sales']
    X = df.drop(columns=['Sales'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Data Split Successful: {len(X_train)} training rows.")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/sales_model.pkl')
    print("Success: Model saved to: models/sales_model.pkl")

if __name__ == "__main__":
    run_training()