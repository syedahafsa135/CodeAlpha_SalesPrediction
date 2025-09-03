# src/train.py
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data import load_and_preprocess_data
import numpy as np

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, r"E:\codealpha1\models\model.pkl")

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2
    }

    print("✅ Model trained and saved successfully!")
    return model, metrics

if __name__ == "__main__":
    train_model()
