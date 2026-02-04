import pandas as pd
import os

def engineer_features():
    input_path = "data/processed/cleaned_sales.csv"
    output_path = "data/processed/engineered_sales.csv"
    
    if not os.path.exists(input_path):
        print("Error: Cleaned data not found.")
        return

    df = pd.read_csv(input_path)
    
    # Create new clues for the AI
    # 1. Profit Margin: Helps the model understand value scale
    df['Profit_Margin'] = df['Profit'] / df['Sales']
    
    # 2. Interaction: Quantity vs Discount
    df['Effective_Quantity'] = df['Quantity'] * (1 - df['Discount'])

    df.to_csv(output_path, index=False)
    print(f"SUCCESS: Engineered features saved to {output_path}")

if __name__ == "__main__":
    engineer_features()