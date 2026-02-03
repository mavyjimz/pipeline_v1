import pandas as pd
import joblib
import os

def run_predictions():
    model_path = 'models/sales_model.pkl'
    # Use the same processed data for a final test prediction
    input_file = 'input_data/processed/sales_summary.csv'
    output_path = 'reports/final_predictions.csv'

    if not os.path.exists(model_path):
        print("Error: Model file not found. Please run training first.")
        return

    model = joblib.load(model_path)
    df = pd.read_csv(input_file)
    
    X = df.drop(columns=['Sales'])
    predictions = model.predict(X)
    
    df['Predicted_Sales'] = predictions
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Success: Predictions saved to {output_path}")

if __name__ == "__main__":
    run_predictions()