import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
INPUT_FILE = r"D:\MLOps\input_data\processed\cleaned_sales.csv"
OUTPUT_FOLDER = r"D:\MLOps\logs" # We will save charts in the logs/reports folder
CHART_FILE = os.path.join(OUTPUT_FOLDER, "sales_by_category.png")

def create_visuals():
    print("📊 Generating Sales Charts...")
    
    # 1. Load the cleaned data
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Group sales by Category
    category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    
    # 3. Create the Chart
    plt.figure(figsize=(10, 6))
    category_sales.plot(kind='bar', color='skyblue', edgecolor='black')
    
    plt.title('Total Sales by Category', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Total Sales ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Save the Chart
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    plt.tight_layout()
    plt.savefig(CHART_FILE)
    print(f"✅ Chart saved successfully to: {CHART_FILE}")

if __name__ == "__main__":
    create_visuals()