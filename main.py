"""
Experiment 16: House Price Prediction System Using Linear Regression
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    data_path = Path(__file__).resolve().parent / "house_price_dataset.csv"
    df = pd.read_csv(data_path)

    print("House Price Dataset Preview:")
    print(df.head())
    print(f"\nDataset rows: {len(df)}")

    feature_columns = [
        "Area", "Bedrooms", "Bathrooms", "Floors",
        "YearBuilt", "Location", "Condition", "Garage"
    ]
    target_column = "Price"

    missing = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")

    X = df[feature_columns]
    y = df[target_column]

    numeric_features = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
    categorical_features = ["Location", "Condition", "Garage"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):,.2f}")
    print(f"R-squared (R²): {r2_score(y_test, y_pred):.4f}")

    sample_house = pd.DataFrame([{
        "Area": 1600,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Floors": 2,
        "YearBuilt": 2010,
        "Location": "Downtown",
        "Condition": "Good",
        "Garage": "Yes",
    }])

    predicted_price = model.predict(sample_house)[0]
    print("\nSample House Details:")
    print(sample_house.to_string(index=False))
    print(f"Predicted Price: {predicted_price:,.2f}")
    print("\nNote: This is an educational estimate, not a guaranteed market price.")


if __name__ == "__main__":
    main()
