
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.database import get_engine
from src.data.baskets import load_basket_summary
from src.features.churn import build_churn_features
from src.models.churn import FEATURE_COLUMNS, predict_churn


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "churn"

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "churn"
)

# Original test configuration
TEST_SNAPSHOT = 622
OBSERVATION_DAYS = 365


def run_churn_inference(
    baskets: pd.DataFrame,
    snapshot_day: int = TEST_SNAPSHOT,
) -> pd.DataFrame:
    """Predict churn using the saved Logistic Regression pipeline."""

    # 1. Load saved model metadata
    with open(
        MODEL_DIR / "churn_model_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    # 2. Load trained model
    model = joblib.load(
        MODEL_DIR / "churn_pipeline.joblib"
    )

    # 3. Read prediction settings
    threshold = float(metadata["threshold"])

    if metadata["features"] != FEATURE_COLUMNS:
        raise ValueError("Model feature configuration mismatch.")

    # 4. Create historical customer features
    customer_features = build_churn_features(
        baskets=baskets,
        snapshot_day=snapshot_day,
        observation_days=OBSERVATION_DAYS,
    )

    # 5. Predict churn without retraining
    predictions = predict_churn(
        model=model,
        customer_features=customer_features,
        threshold=threshold,
    )

    # 6. Validate predictions
    assert predictions["household_key"].is_unique

    assert predictions["churn_probability"].between(
        0, 1
    ).all()

    assert predictions["predicted_churn"].isin([0, 1]).all()

    return predictions


def main() -> None:
    """Run and validate saved-model churn inference."""

    print("CHURN INFERENCE PIPELINE")
    print("-" * 40)

    # 1. Load historical transactions only
    engine = get_engine()

    try:
        baskets = load_basket_summary(
            engine=engine,
            end_day=TEST_SNAPSHOT - 1,
        )
    finally:
        engine.dispose()

    # 2. Generate churn predictions
    predictions = run_churn_inference(
        baskets=baskets,
        snapshot_day=TEST_SNAPSHOT,
    )

    print("\nCHURN PREDICTIONS COMPLETED")
    print("Customers:", len(predictions))

    print("\nPREDICTION DISTRIBUTION")
    print(
        predictions["prediction_label"]
        .value_counts()
        .to_string()
    )

    # 3. Load original training-pipeline predictions
    original = pd.read_csv(
        PROCESSED_DIR / "churn_test_predictions.csv"
    )

    compare_cols = [
        "household_key",
        "churn_probability",
        "predicted_churn",
        "prediction_label",
    ]

    # 4. Align results by customer ID
    original = (
        original[compare_cols]
        .sort_values("household_key")
        .reset_index(drop=True)
    )

    modular = (
        predictions[compare_cols]
        .sort_values("household_key")
        .reset_index(drop=True)
    )

    # 5. Compare customer IDs
    np.testing.assert_array_equal(
        original["household_key"].to_numpy(),
        modular["household_key"].to_numpy(),
    )

    # 6. Compare churn probabilities
    np.testing.assert_allclose(
        original["churn_probability"].to_numpy(),
        modular["churn_probability"].to_numpy(),
        rtol=1e-9,
        atol=1e-9,
    )

    # 7. Compare predicted classifications
    np.testing.assert_array_equal(
        original["predicted_churn"].to_numpy(),
        modular["predicted_churn"].to_numpy(),
    )

    np.testing.assert_array_equal(
        original["prediction_label"].to_numpy(),
        modular["prediction_label"].to_numpy(),
    )

    assert len(modular) == 2499

    print("\nSaved Model Inference: PASSED")
    print("Original Prediction Comparison: PASSED")

    print("\nCHURN INFERENCE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
