import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def train_model():
    input_file = 'input_data/processed/sales_summary.csv'
    model_path = 'models/sales_model.pkl'
    
    df = pd.read_csv(input_file)
    X = df.drop(columns=['Sales'])
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- PHASE 4: EVALUATION ---
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"--- PHASE 4 METRICS ---")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-Squared Score: {r2:.4f}")
    # ---------------------------

    os.makedirs('models', exist_ok=True)
    model_data = {'model': model, 'features': X.columns.tolist()}
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved with updated Phase 4 evaluation.")

if __name__ == "__main__":
    train_model()