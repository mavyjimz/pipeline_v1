import pandas as pd

input_file = 'shared_data/cleaned_sales.csv'
output_file = 'shared_data/engineered_sales.csv'

df = pd.read_csv(input_file)

# New Feature: Average Sales per State (helps the model learn local trends)
df['State_Avg_Sales'] = df.groupby('State')['Sales'].transform('mean')
df['Profit_Margin'] = df['Profit'] / (df['Sales'] + 0.001)

df.to_csv(output_file, index=False)
print(f"Engineered features with State context saved to: {output_file}")