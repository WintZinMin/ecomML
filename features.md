# Feature Inputs and Ranges

- Features used for prediction: `Quantity` (numeric), `Product_Category` (categorical), `Payment_Method` (categorical), `City` (categorical)
- Source: derived from `ecommerce_customer_behavior_dataset_v2.csv` after outlier removal in `run_ml_pipeline.py:174`

## Quantity Range: 1–5
- The saved metadata shows `Quantity.min = 1.0` and `Quantity.max = 5.0` because these are the observed bounds in the training data after outlier filtering.
- Outliers are removed using IQR filtering, which can reduce the observed min/max. See the outlier removal function in `run_ml_pipeline.py:174–184`.
- The input schema is written during training based on the cleaned dataset: `run_ml_pipeline.py:152–170`.
- The app reads this schema and enforces the range for the `Quantity` input: `app.py:41–47`.

## Categorical Values
- Product categories, payment methods, and cities saved in metadata are the unique values present in the cleaned training data.
- Counts are included to indicate coverage:
  - `Product_Category` count: 8
  - `Payment_Method` count: 5
  - `City` count: 10
- The app populates dropdowns from the saved lists ensuring consistency: `app.py:59–66`.

## Why enforce dataset ranges
- Predictions are most reliable within the distribution seen during training. Inputs outside observed ranges can be out-of-distribution and lead to unstable predictions.
- Using the saved schema keeps the UI aligned with the trained model’s domain.

## Metadata Location
- Model: `models/ridge_pipeline.joblib`
- Metadata: `models/ridge_metadata.json` (contains `input_schema` with numeric limits, categorical values, and counts)