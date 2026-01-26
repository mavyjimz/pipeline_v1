import torch
import torch_directml
from gpu_loader import load_production_batch  # From Lesson 27
from predict_sales.py_logic import SalesModel # We will adapt your model here

def run_full_pipeline():
    print("--- [PHASE 1]: DATA LOADING ---")
    # Get our 9,800 rows ready for the GPU
    batch_data = load_production_batch() 
    
    print("\n--- [PHASE 2]: INFERENCE ---")
    # 1. Setup Hardware
    device = torch_directml.device()
    
    # 2. Re-create the Brain (25 inputs)
    model = torch.nn.Sequential(
        torch.nn.Linear(25, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    ).to(device)
    
    # 3. Load weights from Drive D
    model.load_state_dict(torch.load(r"D:\MLOps\models\sales_model.pth", map_location=device))
    model.eval()

    # 4. THE GRAND PREDICTION
    with torch.no_grad():
        results = model(batch_data)
    
    print(f"\n[COMPLETE]: Generated {len(results)} predictions for the Superstore!")
    print(f"Sample Prediction (Row 0): ${results[0].item() * 100:.2f}")

if __name__ == "__main__":
    run_full_pipeline()