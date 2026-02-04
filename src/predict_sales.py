import pandas as pd
import pickle
import os

def generate_predictions():
    model_path = 'models/sales_model.pkl'
    input_file = 'input_data/processed/sales_summary.csv'
    output_file = 'reports/final_predictions.csv'

    print("Starting Phase 3 Dynamic Predictions...")

    if not os.path.exists(model_path):
        print("ERROR: Model not found. Run train_model.py first.")
        return

    # 1. Load Model AND Feature Metadata
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    required_features = model_data['features']

    # 2. Load Processed Data
    df = pd.read_csv(input_file)
    X = df[required_features] # Ensure we use the exact same columns used in training

    # 3. Predict
    predictions = model.predict(X)
    df['Predicted_Sales'] = predictions

    # 4. Save results
    os.makedirs('reports', exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"SUCCESS: Generated predictions for {len(df)} rows using {len(required_features)} features.")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    generate_predictions()