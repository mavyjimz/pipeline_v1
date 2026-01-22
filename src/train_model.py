import pandas as pd
import os
import torch
import torch_directml
import pickle

def train_model():
    # 1. SETUP SYNCED PATHS
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MLOPS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    
    input_file = os.path.join(MLOPS_ROOT, 'data', 'output_data', 'sales_summary.csv')
    model_dir = os.path.join(MLOPS_ROOT, 'models')
    model_file = os.path.join(model_dir, 'sales_predictor.pkl')
    
    os.makedirs(model_dir, exist_ok=True)

    # 2. ENGAGE AMD RX 580
    device = torch_directml.device()
    print(f"\n[AI TRAINING]: Using {torch_directml.device_name(0)}")

    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
        
        # Training logic: Predicting Profit based on Sales
        x_train = torch.tensor(df['Sales'].values, dtype=torch.float32).to(device)
        y_train = torch.tensor(df['Sales'].values, dtype=torch.float32).to(device)

        # Mathematical "Weight" calculation on GPU
        weight = y_train.mean() / x_train.mean()
        
        # 3. SAVE THE MODEL BRAIN
        model_data = {"weight": weight.item(), "hardware": "AMD RX 580"}
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"SUCCESS: AI Model trained and saved to {model_file}")
    else:
        print(f"ERROR: No processed data found at {input_file}")

if __name__ == "__main__":
    train_model()