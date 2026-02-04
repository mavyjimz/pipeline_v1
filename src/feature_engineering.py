import pandas as pd

input_file = 'shared_data/cleaned_sales.csv'
output_file = 'shared_data/engineered_sales.csv'

print("--- Step 2: Feature Engineering ---")
df = pd.read_csv(input_file)

# 1. Derived Feature: Profit Margin
df['Profit_Margin'] = df['Profit'] / df['Sales']

# 2. Derived Feature: Regional Sales Context (Helps Underfitting)
df['Region_Avg_Sales'] = df.groupby('Region')['Sales'].transform('mean')

# Clean up any infinity values from division
df = df.replace([float('inf'), float('-inf')], 0).fillna(0)

df.to_csv(output_file, index=False)
print(f"✓ Success: Created {output_file} with {df.shape[1]} features.")