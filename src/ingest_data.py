import os
import pandas as pd
from pathlib import Path

# --- UNIVERSAL PATH CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "input_data" / "raw" / "superstore_sales.csv"

def sanitize_data():
    if not RAW_FILE.exists():
        print(f"ERROR: Sanitizer cannot find {RAW_FILE}")
        return

    # Load with UTF-8 to handle the superstore_sales.csv errors
    df = pd.read_csv(RAW_FILE, encoding='utf-8', encoding_errors='ignore')
    
    # Remove emojis/special chars from 'Customer Name' and 'Product Name'
    df['Customer Name'] = df['Customer Name'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
    df['Product Name'] = df['Product Name'].str.replace(r'[^\x00-\x7F]+', '', regex=True)
    
    # Overwrite the raw file with the "clean" version
    df.to_csv(RAW_FILE, index=False)
    print(f"SUCCESS: {RAW_FILE} is now sanitized and emoji-free.")

if __name__ == "__main__":
    sanitize_data()