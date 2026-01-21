import pandas as pd
import os
import torch
import torch_directml

def transform_data():
    # Pathing
    input_path = os.path.join("..", "..", "input_data", "raw", "superstore_sales.csv")
    output_path = os.path.join("..", "..", "input_data", "processed", "sales_summary.csv")

    if not os.path.exists(input_path):
        print(f"--- ERROR: {input_path} not found! ---")
        return

    # 1. HARDWARE PROOF: Initialize the AMD RX 580
    device = torch_directml.device()
    gpu_name = torch_directml.device_name(0)
    print(f"\n[HARDWARE CHECK]: Engaging GPU -> {gpu_name}")

    # 2. LOAD DATA
    df = pd.read_csv(input_path)
    df.columns = [str(c).strip() for c in df.columns]
    
    # 3. TENSOR COMPUTATION (Proof of GPU Processing)
    # Moving data from RAM to VRAM
    sales_tensor = torch.tensor(df['Sales'].values, dtype=torch.float32).to(device)
    
    print(f"--- GPU WORKLOAD: Processing {len(sales_tensor)} rows on {gpu_name} ---")
    
    # Mathematical operation performed on the GPU cores
    profit_estimate_tensor = sales_tensor * 0.15
    
    # Moving results back to CPU for CSV saving
    df['Profit'] = profit_estimate_tensor.cpu().numpy()
    df['Profit_Margin'] = 0.15 

    # 4. FINAL AGGREGATION
    if 'Category' in df.columns:
        summary = df.groupby('Category').agg({
            'Sales': 'sum', 
            'Profit': 'sum', 
            'Profit_Margin': 'mean'
        }).reset_index()
    else:
        summary = df[['Sales', 'Profit', 'Profit_Margin']].describe()

   # 5. SAVE AND LOG SUCCESS
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path, index=False)
    
    print("\n" + "*"*45)
    print("   GPU PROOF OF WORK COMPLETE")
    print("*"*45)
    print(summary)
    print("="*45)
    print(f"Verified on Hardware: {gpu_name}\n")

if __name__ == "__main__":
    transform_data()