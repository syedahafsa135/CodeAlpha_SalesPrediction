# src/predict.py
import joblib
import pandas as pd

from src.train import MODEL_PATH


def predict_sales(ad_spend: dict):
    """
    ad_spend = {
        "TV": 230.1,
        "Radio": 37.8,
        "Newspaper": 69.2
    }
    """
    # Load the model when needed, so the app still opens before a model has been trained
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: python -m src.train")
    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([ad_spend])[["TV", "Radio", "Newspaper"]]
    prediction = model.predict(df)[0]
    return round(prediction, 2)

if __name__ == "__main__":
    test_input = {"TV": 150, "Radio": 30, "Newspaper": 20}
    print("Predicted Sales:", predict_sales(test_input))
