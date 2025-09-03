# src/data.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath=r"E:\codealpha1\data\Advertising.csv"):
    # Load dataset
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully. Shape:", df.shape)

    # Drop the index column if present
    if "#" in df.columns or "Unnamed: 0" in df.columns:
        df = df.drop(columns=[df.columns[0]])

    # Define features (X) and target (y)
    X = df.drop("Sales", axis=1)
    y = df["Sales"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data()
