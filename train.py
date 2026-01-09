import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


def main():
    dataset_path = os.path.join("dataset", "winequality-white.csv")  
    output_dir = "output"
    model_path = os.path.join(output_dir, "model.pkl")
    results_path = os.path.join(output_dir, "results.json")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_path, sep=";") 

    # Feature selection & preprocessing
    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Model pipeline (scaling + regression)
    model = Pipeline([
        ("scaler", MinMaxScaler()),
        ("regressor", Lasso(alpha=0.1))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("===== Evaluation Metrics =====")
    print(f"MSE: {mse:.6f}")
    print(f"R2: {r2:.6f}")

    # Save model
    joblib.dump(model, model_path)

    # Save results to JSON
    results = {
        "mse": mse,
        "r2": r2
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
