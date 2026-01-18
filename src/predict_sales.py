import pandas as pd
import joblib
import os

INPUT_FILE = r"D:\MLOps\input_data\processed\cleaned_sales.csv"
MODEL_PATH = r"D:\MLOps\models\sales_model.pkl"

def make_prediction():
    print("🔮 THE ORACLE (Auto-Detection Mode)")
    
    # 1. Load the data to find the categories automatically
    df = pd.read_csv(INPUT_FILE)
    unique_categories = sorted(df['Category'].unique())
    
    # 2. Create the Map automatically (Built to last!)
    cat_map = {name: i for i, name in enumerate(unique_categories)}
    
    print(f"\nI found {len(unique_categories)} categories in your data:")
    print(", ".join(unique_categories))
    
    # 3. Get user input
    choice = input("\nEnter a category: ").title().strip()
    
    if choice in cat_map:
        model = joblib.load(MODEL_PATH)
        prediction = model.predict([[cat_map[choice]]])
        print(f"\n💰 Predicted Sales: ${prediction[0]:.2f}")
    else:
        print("❌ That category doesn't exist in the data!")

if __name__ == "__main__":
    make_prediction()