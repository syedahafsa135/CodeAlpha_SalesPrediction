# src/predict.py
import joblib
import pandas as pd
import os

# Load model
model_path = os.path.join("models", "model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found. Please run train.py first.")

model = joblib.load(model_path)

def predict_sales(ad_spend: dict):
    """
    ad_spend = {
        "TV": 230.1,
        "Radio": 37.8,
        "Newspaper": 69.2
    }
    """
    df = pd.DataFrame([ad_spend])
    prediction = model.predict(df)[0]
    return round(prediction, 2)

if __name__ == "__main__":
    test_input = {"TV": 150, "Radio": 30, "Newspaper": 20}
    print("Predicted Sales:", predict_sales(test_input))
