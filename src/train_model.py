import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

input_file = 'shared_data/engineered_sales.csv'
model_path = 'shared_output/model.joblib'

df = pd.read_csv(input_file)

# Dropping State after using it for the average to prevent 'Overfitting'
X = pd.get_dummies(df.drop(['Sales', 'State'], axis=1))
y = df['Sales']

model = RandomForestRegressor(n_estimators=200, max_depth=25, random_state=42)
model.fit(X, y)

joblib.dump(model, model_path)
print("Model trained on Deep Signal features.")