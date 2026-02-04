import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

input_file = 'shared_data/engineered_sales.csv'
model_path = 'shared_output/model.joblib'

df = pd.read_csv(input_file)

# Encode all categories. We keep State signal inside the 'Avg' feature
X = pd.get_dummies(df.drop(['Sales', 'State'], axis=1))
y = df['Sales']

# MAX CAPACITY: Forcing the model to memorize the fine-grain patterns
model = RandomForestRegressor(
    n_estimators=300, 
    max_depth=35, 
    min_samples_split=2,
    random_state=42
)

model.fit(X, y)
joblib.dump(model, model_path)
print("SUCCESS: High-Capacity model trained on engineered signals.")