import pandas as pd

input_path = "shared_data/cleaned_sales.csv"
output_path = "shared_data/engineered_sales.csv"

print("--- Lesson #558: High-Signal Feature Engineering ---")
df = pd.read_csv(input_path)

# 1. Target Encoding: The average sales for each Sub-Category
# This gives the model a 'baseline' price for every item type
sub_cat_avg = df.groupby('Sub-Category')['Sales'].transform('mean')
df['Item_Value_Index'] = df['Sales'] / (sub_cat_avg + 0.001)

# 2. State-Category Interaction: How does this category sell in this specific state?
state_cat_avg = df.groupby(['State', 'Category'])['Sales'].transform('mean')
df['Local_Demand_Signal'] = state_cat_avg

# 3. Refined Profit Margin
df['Profit_Margin'] = df['Profit'] / (df['Sales'] + 0.001)

df.to_csv(output_path, index=False)
print(f"SUCCESS: High-signal features created in {output_path}")