# Experiment 16: House Price Prediction System Using Linear Regression

## Project Overview
This project demonstrates a machine-learning workflow that predicts house prices using Linear Regression.

## Dataset
`house_price_dataset.csv` contains 2,000 rows and these columns:
- Area
- Bedrooms
- Bathrooms
- Floors
- YearBuilt
- Location
- Condition
- Garage
- Price

The script uses numeric and categorical features. Categorical values are encoded with `OneHotEncoder` in a scikit-learn pipeline.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn

## Project Files
- `main.py` — loads data, trains and evaluates the model, and predicts a sample house price.
- `house_price_dataset.csv` — dataset used by the model.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes virtual environments and cache files.

## Run Locally
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python main.py`

## Evaluation
The script prints Mean Squared Error (MSE) and R-squared (R²) for the held-out test set.

## Important Note
This is an educational experiment. Predictions depend on dataset quality, feature definitions, and price units.
