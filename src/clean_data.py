import pandas as pd

input_file = 'shared_data/superstore_sales.csv'
output_file = 'shared_data/cleaned_sales.csv'

df = pd.read_csv(input_file)

# --- SMART DATA ADAPTATION (Lesson #556) ---
if 'Profit' not in df.columns:
    df['Profit'] = df['Sales'] * 0.15

# EXPANDED FEATURE LIST: Adding Sub-Category and Segment for more "IQ"
categorical_cols = ['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category']
numeric_cols = ['Profit', 'Sales']

# Filter only what we need
df = df[categorical_cols + numeric_cols]

df.to_csv(output_file, index=False)
print(f"Final data saved to: {output_file}")