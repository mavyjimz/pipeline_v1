import pandas as pd
import os
import sys

# ==========================================
# PATH CONFIGURATION (Locked to YAML Bridge)
# ==========================================
# These match the /app/... paths in your docker-compose.yml
INPUT_FILE  = "/app/input_data/processed/cleaned_sales.csv"
OUTPUT_DIR  = "/app/shared_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_predictions.csv")

def diagnostic_check():
    """Verify that the Docker Bridges are actually connected."""
    print("--- SYSTEM DIAGNOSTICS ---")
    print(f"Current Working Directory: {os.getcwd()}")
    
    # Check Input Bridge
    if os.path.exists(INPUT_FILE):
        print(f"Input Bridge: FOUND ({INPUT_FILE})")
    else:
        print(f"Input Bridge: MISSING! Check your YAML ../../input_data mapping.")
        
    # Check/Create Output Bridge
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating Output Directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    else:
        print(f"Output Bridge: READY ({OUTPUT_DIR})")
    print("-----------------------------\n")

def run_pipeline():
    diagnostic_check()
    
    print("Starting Data Processing...")
    try:
        # Load the 9,800 rows
        df = pd.read_csv(INPUT_FILE)
        
        # --- CORE LOGIC ---
        # Simulation of the 5% growth prediction from yesterday
        df['Predicted_Sales'] = df['Sales'] * 1.05
        
        # Save to the Shared Bridge
        df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"SUCCESS: Processed {len(df)} rows.")
        print(f"File saved to: {OUTPUT_FILE}")
        print(f"Check Windows Drive D: D:\\MLOps\\shared_data\\")

    except Exception as e:
        print(f"PIPELINE ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()