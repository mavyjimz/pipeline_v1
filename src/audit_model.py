import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration: Search these paths for prediction artifacts
search_paths = [
    'shared_output/predictions.csv',
    'reports/final_predictions.csv',
    'shared_output/final_predictions.csv',
    'predictions.csv'
]

actuals_path = 'shared_data/engineered_sales.csv'

print("--- Lesson #554: Performance Audit (Automated Path Discovery) ---")

# Search logic to prevent FileNotFoundError
target_path = None
for path in search_paths:
    if os.path.exists(path):
        target_path = path
        break

if not target_path:
    print("CRITICAL ERROR: No prediction file detected. Execute ./src/run_pipeline.sh first.")
else:
    print(f"Data Source Verified: {target_path}")
    
    try:
        # Data ingestion
        y_true = pd.read_csv(actuals_path)['Sales']
        # Dynamically identify the prediction column
        pred_df = pd.read_csv(target_path)
        y_pred = pred_df['Predicted_Sales']

        # Metric Calculation
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print("-" * 40)
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Coefficient of Determination (R2): {r2:.4f}")
        print("-" * 40)
        
        # Diagnostic output
        if r2 < 0.1:
            print("DIAGNOSIS: Underfitting. Signal-to-noise ratio is insufficient.")
        else:
            print("DIAGNOSIS: Signal detected. Model is converging.")

    except Exception as e:
        print(f"RUNTIME ERROR: Analysis failed due to {e}")