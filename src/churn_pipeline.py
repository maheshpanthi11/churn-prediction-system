from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMN = "customerID"
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
RAW_FEATURE_COLUMNS = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create stable, inference-safe business features from raw customer data."""

    def fit(self, features: pd.DataFrame, target: Any = None) -> "ChurnFeatureEngineer":
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        transformed = features.copy()
        transformed["TotalCharges"] = pd.to_numeric(
            transformed["TotalCharges"], errors="coerce"
        )
        transformed["service_count"] = self._service_count(transformed)
        transformed["tenure_group"] = pd.cut(
            pd.to_numeric(transformed["tenure"], errors="coerce"),
            bins=[-1, 6, 12, 24, 48, float("inf")],
            labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49+ months"],
        ).astype("object")
        return transformed

    @staticmethod
    def _service_count(features: pd.DataFrame) -> pd.Series:
        service_columns = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        return features[service_columns].apply(
            lambda column: column.isin(["Yes"]).astype(int)
        ).sum(axis=1)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and minimally normalize the source dataset without fitting anything."""
    dataset = pd.read_csv(csv_path)
    required_columns = {IDENTIFIER_COLUMN, TARGET_COLUMN, *RAW_FEATURE_COLUMNS}
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(str).str.strip()
    return dataset


def build_preprocessor() -> ColumnTransformer:
    numeric_columns = [*NUMERIC_COLUMNS, "service_count"]
    categorical_columns = [*CATEGORICAL_COLUMNS, "tenure_group"]
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    *, max_depth: int | None, min_samples_leaf: int, class_weight: str | None = None
) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", ChurnFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


def split_features_and_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = dataset[RAW_FEATURE_COLUMNS].copy()
    target = dataset[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Churn must contain only 'Yes' or 'No' values")
    return features, target.astype(int)


def get_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())
