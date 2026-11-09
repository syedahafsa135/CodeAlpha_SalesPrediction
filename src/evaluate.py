# src/evaluate.py
# Scores the saved model on the held-out TEST set only (the rows it never saw in training),
# prints the metrics and feature importance, and saves two charts to images/.
#
# Run from the repo folder:  python -m src.evaluate
import joblib
import matplotlib

matplotlib.use("Agg")  # save charts to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import ROOT, load_and_preprocess_data, load_data
from src.train import MODEL_PATH

IMAGES = ROOT / "images"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run: python -m src.train")

    model = joblib.load(MODEL_PATH)
    _, X_test, _, y_test = load_and_preprocess_data()
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("\n📊 Test-set results (held-out rows only):")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {np.sqrt(mse):.2f}")
    print(f"  R²:   {r2_score(y_test, y_pred):.3f}")

    importance = pd.Series(model.feature_importances_, index=X_test.columns).sort_values()
    print("\n📌 Feature importance:")
    for name, value in importance.sort_values(ascending=False).items():
        print(f"  {name}: {value:.3f}")

    IMAGES.mkdir(exist_ok=True)

    # 1. Actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, color="#0E7C86", alpha=0.8)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color="#9AA5B1", linestyle="--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Predicted sales")
    ax.set_title("Actual vs predicted sales (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "actual-vs-predicted.png", dpi=150)

    # 2. Feature importance
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(importance.index, importance.values, color="#0E7C86")
    ax.set_xlabel("Importance")
    ax.set_title("Which advertising channel drives the prediction?")
    fig.tight_layout()
    fig.savefig(IMAGES / "feature-importance.png", dpi=150)

    # 3. Sales against spend, one panel per channel
    df = load_data()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, channel in zip(axes, ["TV", "Radio", "Newspaper"]):
        ax.scatter(df[channel], df["Sales"], color="#0E7C86", alpha=0.6)
        ax.set_xlabel(f"{channel} spend")
        ax.set_title(channel)
    axes[0].set_ylabel("Sales")
    fig.tight_layout()
    fig.savefig(IMAGES / "spend-vs-sales.png", dpi=150)

    print(f"\n🖼️  Charts saved in {IMAGES}")


if __name__ == "__main__":
    main()
