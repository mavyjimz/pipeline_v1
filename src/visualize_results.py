import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_performance():
    input_file = 'reports/final_predictions.csv'
    output_plot = 'reports/performance_chart.png'

    if not os.path.exists(input_file):
        print("ERROR: Run predict_sales.py first.")
        return

    df = pd.read_csv(input_file)

    plt.figure(figsize=(10, 6))
    plt.scatter(df['Sales'], df['Predicted_Sales'], alpha=0.5, color='blue')
    
    # Draw the "Perfect Prediction" line
    max_val = max(df['Sales'].max(), df['Predicted_Sales'].max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--')
    
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Phase 4: Actual vs. Predicted Sales Performance')
    plt.grid(True)
    
    os.makedirs('reports', exist_ok=True)
    plt.savefig(output_plot)
    print(f"SUCCESS: Performance chart saved to {output_plot}")

if __name__ == "__main__":
    plot_performance()