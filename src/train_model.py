from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.churn_pipeline import (
    TARGET_COLUMN,
    build_pipeline,
    get_feature_names,
    load_dataset,
    split_features_and_target,
)


CONFIGURATIONS = {
    "regularized_tree": {"max_depth": 5, "min_samples_leaf": 10, "class_weight": "balanced"},
    "deeper_tree": {"max_depth": 10, "min_samples_leaf": 3, "class_weight": "balanced"},
}


def evaluate(model, features, target) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1_score": f1_score(target, predictions, zero_division=0),
    }


def train(dataset_path: Path, model_path: Path, reports_path: Path) -> dict:
    dataset = load_dataset(dataset_path)
    features, target = split_features_and_target(dataset)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=42,
        stratify=target,
    )

    fitted_models = {}
    results = {}
    for name, configuration in CONFIGURATIONS.items():
        model = build_pipeline(**configuration)
        model.fit(train_features, train_target)
        fitted_models[name] = model
        results[name] = evaluate(model, test_features, test_target)

    selected_name = max(
        results,
        key=lambda name: (results[name]["recall"], results[name]["f1_score"], results[name]["precision"]),
    )
    selected_model = fitted_models[selected_name]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, model_path)

    reports_path.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset_rows": int(len(dataset)),
        "train_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
        "churn_rate": float(target.mean()),
        "selected_model": selected_name,
        "models": results,
        "selected_classification_report": classification_report(
            test_target, selected_model.predict(test_features), target_names=["No", "Yes"], output_dict=True
        ),
    }
    (reports_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = confusion_matrix(test_target, selected_model.predict(test_features))
    ConfusionMatrixDisplay(matrix, display_labels=["No", "Yes"]).plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {selected_name}")
    plt.tight_layout()
    plt.savefig(reports_path / "confusion_matrix.png", dpi=150)
    plt.close()

    feature_importance = pd.DataFrame(
        {
            "feature": get_feature_names(selected_model),
            "importance": selected_model.named_steps["classifier"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(reports_path / "feature_importance.csv", index=False)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Telco churn decision-tree pipeline")
    parser.add_argument("--data", type=Path, default=Path("data/TelcoCustomerChurn.csv"))
    parser.add_argument("--model", type=Path, default=Path("model/churn_model.pkl"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    arguments = parser.parse_args()
    print(json.dumps(train(arguments.data, arguments.model, arguments.reports), indent=2))


if __name__ == "__main__":
    main()
