import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, r2_score

# Resilience: Check multiple artifact locations
targets = ['reports/final_predictions.csv', 'shared_output/predictions.csv']
actuals = 'shared_data/engineered_sales.csv'

print("--- PIPELINE PERFORMANCE AUDIT (Lesson #560) ---")

active_path = next((p for p in targets if os.path.exists(p)), None)

if not active_path:
    print("FATAL: Prediction artifacts not found. Run pipeline first.")
else:
    y_true = pd.read_csv(actuals)['Sales']
    y_pred = pd.read_csv(active_path)['Predicted_Sales']
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"Artifact Source: {active_path}")
    print("-" * 30)
    print(f"R2 SCORE:  {r2:.4f}")
    print(f"AVG ERROR: ${mae:.2f}")
    print("-" * 30)
    
    if r2 > 0.0503:
        print("RESULT: Plateau Broken. Information Ceiling bypassed.")
    else:
        print("RESULT: Stagnant. Dataset requires external feature injection.")