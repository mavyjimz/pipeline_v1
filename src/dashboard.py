import pandas as pd
import json
import os

def generate_dashboard_metadata():
    input_file = 'reports/final_predictions.csv'
    output_json = 'reports/dashboard_metrics.json'
    
    print("--- Lesson #570: Synchronizing Phase 3 JSON Bridge ---")
    
    if not os.path.exists(input_file):
        print("ERROR: CSV data missing. Run pipeline first.")
        return

    df = pd.read_csv(input_file)
    
    # Extract telemetry for Chrome presentation
    # Note: Using the R2/MAE from our latest high-complexity audit
    metadata = {
        "project_name": "Superstore MLOps Pipeline",
        "model_version": "v1.5_HighComplexity",
        "performance_metrics": {
            "r2_score": 0.0503,
            "mean_absolute_error": 238.02,
            "status": "Stagnant/Underfitting"
        },
        "data_summary": {
            "total_rows": len(df),
            "top_predicted_sale": float(df['Predicted_Sales'].max())
        },
        "visual_artifacts": {
            "chart_path": "reports/performance_chart.png"
        }
    }

    with open(output_json, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("SUCCESS: JSON metadata synchronized for Chrome visibility at " + output_json)

if __name__ == "__main__":
    generate_dashboard_metadata()