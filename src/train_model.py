import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

input_file = 'shared_data/engineered_sales.csv'
model_path = 'shared_output/model.joblib'

print("--- Step 3: Training Model ---")
df = pd.read_csv(input_file)

# Encode categorical data
df_encoded = pd.get_dummies(df, columns=['Ship Mode', 'Segment', 'Region', 'Category'])

# Define features (Matches our Engineering Step)
X = df_encoded.drop(['Sales'], axis=1)
y = df_encoded['Sales']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, model_path)
print(f"✓ Success: Model saved to {model_path}")