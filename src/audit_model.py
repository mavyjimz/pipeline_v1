import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Paths
actuals_path = 'shared_data/engineered_sales.csv'
preds_path = 'shared_output/predictions.csv'

print("--- Lesson #550: Model Performance Audit ---")

# Load data
y_true = pd.read_csv(actuals_path)['Sales']
y_pred = pd.read_csv(preds_path)['Predicted_Sales']

# Calculate Metrics
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Average Error: ${mae:.2f}")
print(f"Model Accuracy (R2 Score): {r2:.4f}")

if r2 < 0.5:
    print("STATUS: Underfitting Detected. Model is too simple.")
else:
    print("STATUS: Model is learning! Ready for tuning.")