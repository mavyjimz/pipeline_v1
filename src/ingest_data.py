import pandas as pd
import os

def ingest_data():
    # SETUP PATHS TO MLOPS ROOT (3 levels up from src)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MLOPS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    
    DATA_DIR = os.path.join(MLOPS_ROOT, 'data', 'input_data')
    os.makedirs(DATA_DIR, exist_ok=True)
    
    file_path = os.path.join(DATA_DIR, 'raw_data.csv')
    
    # Create sample data
    data = {
        'Date': pd.date_range(start='2025-01-01', periods=10, freq='D'),
        'Sales': [100, 150, 120, 200, 180, 250, 220, 300, 280, 350],
        'Temperature': [22, 24, 21, 25, 26, 28, 27, 30, 29, 31]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"SUCCESS: Data ingested to {file_path}")

if __name__ == "__main__":
    ingest_data()