
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.database import get_engine
from src.data.baskets import load_basket_summary
from src.features.churn import build_churn_snapshot_dataset
from src.models.churn import (
    FEATURE_COLUMNS,
    fit_churn_model,
    predict_churn,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "churn"
)

MODEL_DIR = PROJECT_ROOT / "models" / "churn"

# Time-based configuration
TRAIN_SNAPSHOTS = [366, 396, 426]
VALIDATION_SNAPSHOT = 516
TEST_SNAPSHOT = 622

OBSERVATION_DAYS = 365
TARGET_DAYS = 90


def main() -> None:
    """Run the customer churn training pipeline."""

    print("CUSTOMER CHURN TRAINING PIPELINE")
    print("-" * 40)

    # 1. Create output directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Validate time-based splits
    assert (
        max(TRAIN_SNAPSHOTS) + TARGET_DAYS
        <= VALIDATION_SNAPSHOT
    )

    assert (
        VALIDATION_SNAPSHOT + TARGET_DAYS
        <= TEST_SNAPSHOT
    )

    # 3. Load basket-level data
    engine = get_engine()

    try:
        basket_df = load_basket_summary(
            engine=engine,
            end_day=711,
        )
    finally:
        engine.dispose()

    print("\nDATA LOADING COMPLETED")
    print("Total Baskets:", len(basket_df))

    # 4. Build training dataset
    train_df = pd.concat(
        [
            build_churn_snapshot_dataset(
                baskets=basket_df,
                snapshot_day=day,
                observation_days=OBSERVATION_DAYS,
                target_days=TARGET_DAYS,
            )
            for day in TRAIN_SNAPSHOTS
        ],
        ignore_index=True,
    )

    # 5. Build validation dataset
    val_df = build_churn_snapshot_dataset(
        baskets=basket_df,
        snapshot_day=VALIDATION_SNAPSHOT,
        observation_days=OBSERVATION_DAYS,
        target_days=TARGET_DAYS,
    )

    # 6. Build test dataset
    test_df = build_churn_snapshot_dataset(
        baskets=basket_df,
        snapshot_day=TEST_SNAPSHOT,
        observation_days=OBSERVATION_DAYS,
        target_days=TARGET_DAYS,
    )

    # 7. Check dataset distribution
    dataset_summary = pd.DataFrame({
        "dataset": ["Train", "Validation", "Test"],
        "rows": [
            len(train_df),
            len(val_df),
            len(test_df),
        ],
        "churn_count": [
            train_df["churn_90d"].sum(),
            val_df["churn_90d"].sum(),
            test_df["churn_90d"].sum(),
        ],
    })

    dataset_summary["churn_rate_pct"] = (
        dataset_summary["churn_count"]
        / dataset_summary["rows"]
        * 100
    )

    print("\nDATASET PREPARATION COMPLETED")
    print(dataset_summary.round(2).to_string(index=False))

    # 8. Train the selected classification model
    churn_model = fit_churn_model(
        train_df=train_df,
    )

    print("\nMODEL TRAINING COMPLETED")
    print("Model: Balanced Logistic Regression")
    print("Training Samples:", len(train_df))

    # 9. Load original model and metadata
    original_model = joblib.load(
        PROJECT_ROOT
        / "models"
        / "churn_logistic_pipeline.joblib"
    )

    with open(
        PROJECT_ROOT
        / "models"
        / "churn_model_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    threshold = float(metadata["threshold"])

    assert metadata["features"] == FEATURE_COLUMNS

    # 10. Compare original and modular models
    X_val = val_df[FEATURE_COLUMNS]

    original_proba = original_model.predict_proba(
        X_val
    )[:, 1]

    modular_proba = churn_model.predict_proba(
        X_val
    )[:, 1]

    np.testing.assert_allclose(
        original_proba,
        modular_proba,
        rtol=1e-6,
        atol=1e-6,
    )

    print("\nOriginal Model Comparison: PASSED")

    # 11. Generate validation predictions
    val_predictions = predict_churn(
        model=churn_model,
        customer_features=val_df,
        threshold=threshold,
    )

    y_true = val_df["churn_90d"].to_numpy()

    y_proba = val_predictions[
        "churn_probability"
    ].to_numpy()

    y_pred = val_predictions[
        "predicted_churn"
    ].to_numpy()

    # 12. Evaluate validation performance
    validation_metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(
            y_true, y_proba
        ),
        "f1": f1_score(
            y_true, y_pred,
            zero_division=0,
        ),
    }

    print("\nVALIDATION RESULTS")
    print("Threshold:", threshold)

    for metric, value in validation_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nChurn Training Validation: PASSED")

    # 13. Generate predictions on the final test dataset
    test_predictions = predict_churn(
        model=churn_model,
        customer_features=test_df,
        threshold=threshold,
    )

    # 14. Prepare actual and predicted values
    y_test = test_df["churn_90d"].to_numpy()

    y_test_proba = test_predictions[
        "churn_probability"
    ].to_numpy()

    y_test_pred = test_predictions[
        "predicted_churn"
    ].to_numpy()

    # 15. Calculate test metrics
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(
            y_test, y_test_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_test_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_test_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_test_proba),
        "average_precision": average_precision_score(
            y_test, y_test_proba
        ),
    }

    # 16. Calculate confusion matrix
    cm = confusion_matrix(
        y_test,
        y_test_pred,
        labels=[0, 1],
    )

    # 17. Display results
    print("\nFINAL TEST EVALUATION COMPLETED")
    print("Test Customers:", len(test_df))
    print("Classification Threshold:", threshold)

    print("\nTEST METRICS")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nCONFUSION MATRIX")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Active", "Actual Inactive"],
        columns=["Predicted Active", "Predicted Inactive"],
    )

    print(cm_df.to_string())

    print("\nFinal Test Validation: PASSED")

    # 18. Prepare customer-level test predictions
    test_results = test_df[
        ["household_key", "snapshot_day", "churn_90d"]
    ].reset_index(drop=True).copy()

    np.testing.assert_array_equal(
        test_results["household_key"].to_numpy(),
        test_predictions["household_key"].to_numpy(),
    )

    test_results["churn_probability"] = y_test_proba
    test_results["predicted_churn"] = y_test_pred
    test_results["prediction_label"] = (
        test_predictions["prediction_label"].to_numpy()
    )

    # 19. Define artifact paths
    artifact_paths = {
        "model": MODEL_DIR / "churn_pipeline.joblib",
        "metadata": MODEL_DIR / "churn_model_metadata.json",
        "predictions": (
            PROCESSED_DIR / "churn_test_predictions.csv"
        ),
        "metrics": (
            PROCESSED_DIR / "churn_evaluation_metrics.json"
        ),
    }

    # 20. Save trained pipeline
    joblib.dump(
        churn_model,
        artifact_paths["model"],
    )

    # 21. Save test predictions
    test_results.to_csv(
        artifact_paths["predictions"],
        index=False,
    )

    # 22. Prepare model metadata
    model_metadata = {
        **metadata,
        "model_name": "Balanced Logistic Regression",
        "features": FEATURE_COLUMNS,
        "threshold": float(threshold),
        "training_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "validation_metrics": {
            key: float(value)
            for key, value in validation_metrics.items()
        },
        "test_metrics": {
            key: float(value)
            for key, value in test_metrics.items()
        },
    }

    # 23. Save metadata
    with open(
        artifact_paths["metadata"],
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # 24. Save evaluation results
    evaluation_results = {
        "model": "Balanced Logistic Regression",
        "threshold": float(threshold),
        "validation": model_metadata["validation_metrics"],
        "test": model_metadata["test_metrics"],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [0, 1],
    }

    with open(
        artifact_paths["metrics"],
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation_results,
            file,
            indent=4,
        )

    # 25. Reload model and verify predictions
    loaded_model = joblib.load(
        artifact_paths["model"]
    )

    reloaded_proba = loaded_model.predict_proba(
        test_df[FEATURE_COLUMNS]
    )[:, 1]

    np.testing.assert_allclose(
        y_test_proba,
        reloaded_proba,
        rtol=1e-9,
        atol=1e-9,
    )

    # 26. Validate saved files
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name} artifact: {path}"
            )

    print("\nARTIFACT SAVING COMPLETED")
    print("-" * 40)

    for name, path in artifact_paths.items():
        print(f"{name}: {path}")

    print("\nModel Reload Check: PASSED")
    print("Artifact Validation: PASSED")

    print("\nCHURN TRAINING PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
