# src/data.py
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Paths are worked out from this file's location, so the project runs from any folder or computer
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Advertising.csv"


def load_data(filepath=DATA_PATH):
    """Load the Advertising dataset and drop the unnamed index column if present."""
    df = pd.read_csv(filepath)
    if "#" in df.columns or "Unnamed: 0" in df.columns:
        df = df.drop(columns=[df.columns[0]])
    return df


def load_and_preprocess_data(filepath=DATA_PATH):
    df = load_data(filepath)
    print("Dataset loaded successfully. Shape:", df.shape)

    # Features (X) and target (y)
    X = df.drop("Sales", axis=1)
    y = df["Sales"]

    # 80/20 train-test split (fixed seed so results are repeatable)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_preprocess_data()
