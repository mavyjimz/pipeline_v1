import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

input_file = 'shared_data/engineered_sales.csv'
model_path = 'shared_output/model.joblib'

print("--- Lesson #555: Training High-Complexity Model ---")
df = pd.read_csv(input_file)

# One-hot encoding categories
df_encoded = pd.get_dummies(df)

X = df_encoded.drop(['Sales'], axis=1)
y = df_encoded['Sales']

# INCREASING IQ: Adding more trees and allowing them to grow deeper
model = RandomForestRegressor(
    n_estimators=200,    # Doubled the trees
    max_depth=20,        # Allowed more complex logic
    min_samples_split=5, # Prevents over-simplifying
    random_state=42
)

model.fit(X, y)

joblib.dump(model, model_path)
print(f"Success: High-IQ Model saved to {model_path}")