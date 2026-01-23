import torch
import torch.nn as nn
import os

# 1. THE BRAIN STRUCTURE (Must match the new training math)
class SalesPredictor(nn.Module):
    def __init__(self):
        super(SalesPredictor, self).__init__()
        self.layer = nn.Linear(3, 1) # Must be 3 for Lesson 14!

    def forward(self, x):
        return self.layer(x)

def predict():
    model_path = "models/sales_predictor.pkl"
    if not os.path.exists(model_path):
        print("ERROR: The Oracle is silent. Run the pipeline first!")
        return

    # Load the model we just trained
    model = torch.load(model_path)
    model.eval()

    print("\n--- THE ORACLE OF SALES (MULTI-FEATURE) ---")
    try:
        # Now we collect all 3 pieces of the puzzle
        print("Please enter the numeric IDs:")
        cat = float(input("1. Category ID (e.g., 0, 1, 2): "))
        reg = float(input("2. Region ID (e.g., 0, 1, 2, 3): "))
        seg = float(input("3. Segment ID (e.g., 0, 1, 2): "))
        
        # We put them in a list-of-lists to create a 1x3 matrix
        input_data = torch.tensor([[cat, reg, seg]], dtype=torch.float32)
        
        with torch.no_grad():
            prediction = model(input_data).item()
        
        print(f"\n🔮 PREDICTED SALES: ${prediction:.2f}")
        print("------------------------------------------")
    
    except ValueError:
        print("ERROR: Please enter numbers only for the IDs.")

if __name__ == "__main__":
    predict()