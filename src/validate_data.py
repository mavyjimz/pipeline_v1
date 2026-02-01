import pandas as pd
import os

def validate_sales_data(file_path):
    print("LOG: Starting validation on: " + file_path)
    
    if not os.path.exists(file_path):
        print("ERROR: File not found.")
        return

    try:
        df = pd.read_csv(file_path)
        
        # The Core Check
        if 'Sales' not in df.columns:
            print("ALARM: Missing 'Sales' column!")
        else:
            print("SUCCESS: 'Sales' column verified.")
            print(df.head())
            
    except Exception as e:
        print("ERROR: Could not read CSV. Details: " + str(e))

if __name__ == "__main__":
    # Drive D project path
    target_file = r'D:\MLOps\projects\pipeline_v1\input_data\superstore_sales.csv'
    validate_sales_data(target_file)