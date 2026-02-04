import pandas as pd
import os

# Paths synced with Docker Volume Bridge
input_file = 'shared_data/superstore_sales.csv'
output_file = 'shared_data/cleaned_sales.csv'

print("--- Step 1: Cleaning Data ---")
df = pd.read_csv(input_file)

# Logic to handle the 'notorious' missing columns
if 'Profit' not in df.columns:
    print("! Warning: Profit missing. Simulating 15% margin.")
    df['Profit'] = df['Sales'] * 0.15

# Keep only what we verified exists + our simulated Profit
categorical_cols = ['Ship Mode', 'Segment', 'Region', 'Category']
numeric_cols = ['Sales', 'Profit'] 

df_cleaned = df[categorical_cols + numeric_cols].dropna()

df_cleaned.to_csv(output_file, index=False)
print(f"✓ Success: Saved to {output_file}")