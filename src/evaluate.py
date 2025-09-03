# src/evaluate.py
import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

data_path = os.path.join("data", "Advertising.csv")
model_path = os.path.join("models", "model.pkl")

# Load dataset
df = pd.read_csv(data_path)

# Drop index column if present
if "#" in df.columns or "Unnamed: 0" in df.columns:
    df = df.drop(columns=[df.columns[0]])

X = df.drop("Sales", axis=1)
y = df["Sales"]

# Load trained model
model = joblib.load(model_path)

# Predict
y_pred = model.predict(X)

# Metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
