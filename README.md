# Telco Customer Churn Prediction

An end-to-end customer churn solution using the IBM Telco Customer Churn dataset. It includes data preparation, EDA, feature engineering, Decision Tree comparison, model interpretation, a saved preprocessing/model pipeline, and a Flask prediction API.

## Project Structure

```text
.
├── TelcoCustomerChurn.csv
├── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/churn_analysis.ipynb
├── src/churn_pipeline.py
├── src/train_model.py
├── model/churn_model.pkl
├── reports/
├── app.py
├── sample_request.json
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Train the Model

Run from the project root:

```bash
python3 -m src.train_model \
  --data TelcoCustomerChurn.csv \
  --model model/churn_model.pkl \
  --reports reports
```

The training script uses a stratified 70:30 train/test split and `random_state=42`. It compares two Decision Trees and saves the selected complete pipeline, including feature engineering, imputation, and one-hot encoding. The current selection criterion prioritizes recall, then F1 and precision, because missing a likely churner is usually more costly than contacting an additional low-risk customer.

Generated reports include `metrics.json`, `feature_importance.csv`, and `confusion_matrix.png`.

## Notebook

Launch Jupyter from the project root:

```bash
jupyter notebook
```

Open `notebook/churn_analysis.ipynb` and run all cells. The notebook documents data quality, EDA, business insights, feature engineering, model comparison, evaluation, and interpretation.

## API

Start the API after generating the model:

```bash
python3 app.py
```

Health check:

```bash
curl http://localhost:5000/health
```

Prediction request:

```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  --data @sample_request.json
```

Example response:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```

`POST /predict` requires the customer feature fields shown in `sample_request.json`. `customerID` is intentionally excluded because it is an identifier and is not used for prediction. Invalid JSON, missing fields, invalid numeric fields, and incorrect field types return HTTP 400.

## Data and Modeling Notes

- Blank `TotalCharges` values are converted to missing numeric values and median-imputed inside the fitted pipeline.
- `customerID` is excluded from model features to avoid learning identifier noise.
- `tenure_group` captures meaningful customer lifecycle bands: 0-6, 7-12, 13-24, 25-48, and 49+ months.
- `service_count` counts subscribed services marked `Yes`, which summarizes engagement breadth.
- One-hot encoding uses `handle_unknown="ignore"`, allowing unseen categorical values at inference time without breaking the API.
- The dataset contains 7,043 rows, no duplicate rows, 11 blank `TotalCharges` values, and approximately 26.54% churn.

## Limitations

This is an educational predictive model, not a causal churn model. The default classification threshold is 0.5, and a production retention workflow should tune it using intervention capacity and the relative costs of false positives and false negatives. Model performance should also be monitored on newer customer cohorts.
