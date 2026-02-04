import pandas as pd

input_file = 'shared_data/cleaned_sales.csv'
output_file = 'shared_data/engineered_sales.csv'

df = pd.read_csv(input_file)

# 1. State-Product Interaction (The 'Local Price' signal)
df['State_SubCat_Avg'] = df.groupby(['State', 'Sub-Category'])['Sales'].transform('mean')

# 2. Relative Value Index
df['Value_Ratio'] = df['Sales'] / (df['State_SubCat_Avg'] + 0.001)

# 3. Profit Efficiency
df['Profit_Margin'] = df['Profit'] / (df['Sales'] + 0.001)

df.to_csv(output_file, index=False)
print("SUCCESS: Localized market signals generated.")