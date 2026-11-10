
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.database import get_engine
from src.data.baskets import load_basket_summary
from src.features.cltv import prepare_purchase_days
from src.models.customer_value import predict_customer_revenue


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "crm"

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "crm"
)


def run_crm_inference(
    baskets: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Generate CRM forecasts using a saved BG-NBD model."""

    # 1. Load model metadata
    metadata_path = MODEL_DIR / "crm_model_metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # 2. Load trained model
    model_path = MODEL_DIR / "bgnbd_model.joblib"
    model = joblib.load(model_path)

    # 3. Read model parameters
    snapshot_day = int(metadata["snapshot_day"])
    training_days = int(metadata["training_days"])
    horizon = int(metadata["forecast_horizon_days"])

    # 4. Prepare purchase-day features
    purchase_days = prepare_purchase_days(
        baskets=baskets,
        snapshot_day=snapshot_day,
        lookback_days=training_days,
    )

    # 5. Generate predictions without retraining
    predictions, corrected_count = predict_customer_revenue(
        model=model,
        purchase_days=purchase_days,
        snapshot_day=snapshot_day,
        horizon=horizon,
    )

    return predictions, corrected_count


def main() -> None:
    """Run and validate CRM batch inference."""

    print("CRM INFERENCE PIPELINE")
    print("-" * 40)

    # 1. Load basket data
    engine = get_engine()

    try:
        basket_df = load_basket_summary(engine)
    finally:
        engine.dispose()

    # 2. Predict using saved model
    predictions, corrected_count = run_crm_inference(
        baskets=basket_df,
    )

    # 3. Load saved training pipeline predictions
    original = pd.read_csv(
        PROCESSED_DIR / "customer_revenue_predictions.csv"
    )

    # 4. Compare results by customer ID
    compare_cols = [
        "expected_purchases_90d",
        "historical_monetary",
        "expected_revenue_90d",
    ]

    original = (
        original
        .sort_values("household_key")
        .reset_index(drop=True)
    )

    predictions = (
        predictions
        .sort_values("household_key")
        .reset_index(drop=True)
    )

    np.testing.assert_array_equal(
        original["household_key"].to_numpy(),
        predictions["household_key"].to_numpy(),
    )

    np.testing.assert_allclose(
        original[compare_cols].to_numpy(dtype=float),
        predictions[compare_cols].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-6,
    )

    print("Customers:", len(predictions))
    print("Corrected Predictions:", corrected_count)

    print("\nSaved Model Inference: PASSED")
    print("Original Prediction Comparison: PASSED")

    print("\nCRM INFERENCE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
