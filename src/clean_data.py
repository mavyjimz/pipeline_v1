import pandas as pd
import os

def clean_data():
    # SETUP PATHS TO MLOPS ROOT (3 levels up from src)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MLOPS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    
    input_file = os.path.join(MLOPS_ROOT, 'data', 'input_data', 'raw_data.csv')
    output_dir = os.path.join(MLOPS_ROOT, 'data', 'output_data')
    output_file = os.path.join(output_dir, 'cleaned_data.csv')

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        
        df.to_csv(output_file, index=False)
        print(f"SUCCESS: Data cleaned and saved to {output_file}")
    else:
        print(f"ERROR: Could not find {input_file}")

if __name__ == "__main__":
    clean_data()