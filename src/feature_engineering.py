import pandas as pd
import os

def engineer_features():
    # DIRECTORY ALIGNMENT - Using the shared_data folder we created
    input_path = "shared_data/cleaned_sales.csv"
    output_path = "shared_data/engineered_sales.csv"
    
    # Safety Check: Stop if the previous step failed
    if not os.path.exists(input_path):
        print(f"ERROR: Could not find {input_path}")
        print("Check if Lesson #532 (Cleaning) was successful in the shared_data folder.")
        return

    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # --- FEATURE ENGINEERING (The 'Phase 5' Magic) ---
    
    # 1. Profit Margin: Helps AI understand the relationship between cost and revenue
    # We add a small 0.001 to Sales to prevent "Division by Zero" errors
    df['Profit_Margin'] = df['Profit'] / (df['Sales'] + 0.001)
    
    # 2. Performance Metric: Interaction between Quantity and Discount
    df['Discount_Impact'] = df['Quantity'] * df['Discount']

    # Save to the shared volume so the next script can see it
    df.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"SUCCESS: {output_path} created!")
    print(f"New Columns Added: ['Profit_Margin', 'Discount_Impact']")
    print("-" * 30)

if __name__ == "__main__":
    engineer_features()