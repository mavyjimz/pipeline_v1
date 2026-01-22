import torch
import torch.nn as nn
import os

# --- 1. THE BLUEPRINT ---
# This MUST match the class in train_model.py so Python can rebuild the model
class SalesModel(nn.Module):
    def __init__(self):
        super(SalesModel, self).__init__()
        self.layer = nn.Linear(1, 1) 
    def forward(self, x):
        return self.layer(x)

# --- 2. CONFIGURATION ---
MODEL_PATH = r"D:\MLOps\models\sales_predictor.pkl"

def run_oracle():
    print("🔮 --- The Sales Oracle is Online ---")
    
    # Check if the file exists at the correct path
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    # Map the categories to numerical values
    cat_map = {
        "1": {"name": "Furniture", "val": 0.0},
        "2": {"name": "Office Supplies", "val": 1.0},
        "3": {"name": "Technology", "val": 2.0}
    }

    print("\nSelect a Category to Predict:")
    for key, info in cat_map.items():
        print(f"[{key}] {info['name']}")

    choice = input("\nEnter choice (1-3): ")

    if choice in cat_map:
        try:
            # 3. LOAD THE MODEL
            # We use weights_only=False because we are loading the full object class
            model = torch.load(MODEL_PATH, weights_only=False)
            model.eval() # Set to evaluation mode

            # 4. PREPARE INPUT & PREDICT
            category_value = cat_map[choice]['val']
            # Convert to a 2D tensor [[val]]
            input_tensor = torch.tensor([[category_value]], dtype=torch.float32)
            
            with torch.no_grad():
                prediction = model(input_tensor)

            # 5. DISPLAY RESULTS
            print("-" * 35)
            print(f"✅ TARGET: {cat_map[choice]['name']}")
            print(f"💰 PREDICTED SALES: ${prediction.item():,.2f}")
            print("-" * 35)

        except Exception as e:
            print(f"❌ Oracle Error: {e}")
            print("💡 Hint: Try re-running 'python src/train_model.py' first.")
    else:
        print("Invalid selection. Oracle shutting down.")

if __name__ == "__main__":
    run_oracle()