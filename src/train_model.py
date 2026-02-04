import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_model():
    input_file = 'input_data/processed/sales_summary.csv'
    model_path = 'models/sales_model.pkl'
    
    print("Starting Phase 3 Dynamic Training...")

    if not os.path.exists(input_file):
        print("ERROR: Processed data not found. Run clean_data.py first.")
        return

    df = pd.read_csv(input_file)

    # 1. DYNAMIC FEATURE SELECTION
    # We take EVERY column except 'Sales' as a feature
    X = df.drop(columns=['Sales'])
    y = df['Sales']

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data Split Success: {len(X_train)} training rows | {X_train.shape[1]} features.")

    # 3. Train the Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Save the Model and the list of Feature Names
    # We save the feature names so the Predictor knows what to expect!
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': model,
        'features': X.columns.tolist()
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"SUCCESS: Model and {len(X.columns)} feature names saved to {model_path}")

if __name__ == "__main__":
    train_model()