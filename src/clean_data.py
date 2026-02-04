import pandas as pd
import os

def clean_data():
    input_file = 'input_data/raw/superstore_sales.csv'
    output_file = 'input_data/processed/sales_summary.csv'
    
    print("Starting Phase 3 Cleaning & Encoding...")

    # Load the data with the encoding we verified earlier
    try:
        df = pd.read_csv(input_file, encoding='latin1')
    except Exception as e:
        print(f"ERROR: Could not read file. {e}")
        return

    # 1. Feature Selection: Choosing the variables that drive sales
    # We include Categorical data now!
    categorical_cols = ['Ship Mode', 'Segment', 'Region', 'Category']
    target_col = 'Sales'
    
    # Keep only the columns we need
    df = df[categorical_cols + [target_col]]

    # 2. Handle Missing Values
    initial_rows = len(df)
    df = df.dropna()
    print(f"Cleaned {initial_rows - len(df)} empty rows.")

    # 3. One-Hot Encoding (The Magic Step)
    # This turns text into binary (0 and 1) so the AI can understand it
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # 4. Save the results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_encoded.to_csv(output_file, index=False)
    
    print(f"SUCCESS: Processed {df_encoded.shape[1]} total features.")
    print(f"Final data saved to: {output_file}")

if __name__ == "__main__":
    clean_data()