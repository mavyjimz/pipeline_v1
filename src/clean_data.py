import pandas as pd

input_file = 'shared_data/superstore_sales.csv'
output_file = 'shared_data/cleaned_sales.csv'

df = pd.read_csv(input_file)

if 'Profit' not in df.columns:
    df['Profit'] = df['Sales'] * 0.15

# DEEP FEATURES: State and Product ID give the model specific 'anchors'
categorical_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category', 'State']
numeric_cols = ['Profit', 'Sales']

df = df[categorical_cols + numeric_cols].dropna()
df.to_csv(output_file, index=False)
print(f"Deep data saved to: {output_file}")