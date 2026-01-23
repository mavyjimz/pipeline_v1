import pandas as pd
import os

def clean_data():
    # 1. DEFINE ROOTS (Consistent with Suspect #1)
    MLOPS_ROOT = r"D:\MLOps"
    
    # 2. ALIGN WITH OFFICIAL WAREHOUSE
    # Input comes from 'raw', Output goes to 'processed'
    input_file = os.path.join(MLOPS_ROOT, 'input_data', 'raw', 'raw_data.csv')
    output_dir = os.path.join(MLOPS_ROOT, 'input_data', 'processed')
    output_file = os.path.join(output_dir, 'sales_summary.csv')

    # Create the standard directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        
        # Perform cleaning
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        
        # Save to the OFFICIAL "Certified Clean" location
        df.to_csv(output_file, index=False)
        print(f"SUCCESS: Data cleaned and saved to {output_file}")
    else:
        print(f"ERROR: Could not find {input_file}")

if __name__ == "__main__":
    clean_data()