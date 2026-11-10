
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data.database import get_engine
from src.data.baskets import load_basket_summary
from src.features.segmentation import build_segmentation_features
from src.models.segmentation import integrate_rfm_kmeans


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "segmentation"

SEGMENTATION_DIR = (
    PROJECT_ROOT / "data" / "processed" / "segmentation"
)

CRM_DIR = PROJECT_ROOT / "data" / "processed" / "crm"


def run_segmentation_inference(
    baskets: pd.DataFrame,
) -> pd.DataFrame:
    """Predict customer clusters using a saved K-Means pipeline."""

    # 1. Load model metadata
    metadata_path = MODEL_DIR / "segmentation_metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    # 2. Load trained pipeline
    model = joblib.load(
        MODEL_DIR / "kmeans_pipeline.joblib"
    )

    # 3. Read training configuration
    snapshot_day = int(metadata["snapshot_day"])
    lookback_days = int(metadata["lookback_days"])
    feature_cols = metadata["selected_features"]

    cluster_names = {
        int(key): value
        for key, value in metadata["cluster_names"].items()
    }

    # 4. Rebuild customer features
    customer_features = build_segmentation_features(
        baskets=baskets,
        snapshot_day=snapshot_day,
        lookback_days=lookback_days,
    )

    # 5. Predict clusters without retraining
    predictions = model.predict(
        customer_features[feature_cols]
    )

    # 6. Assign cluster labels
    results = customer_features[["household_key"]].copy()

    results["cluster"] = predictions

    results["cluster_name"] = (
        results["cluster"].map(cluster_names)
    )

    # 7. Validate predictions
    if results["cluster_name"].isna().any():
        raise ValueError("Unknown cluster labels found.")

    if not results["household_key"].is_unique:
        raise ValueError("Duplicate customers found.")

    return results


def main() -> None:
    """Run and validate segmentation inference."""

    print("SEGMENTATION INFERENCE PIPELINE")
    print("-" * 40)

    # 1. Load metadata for the observation window
    with open(
        MODEL_DIR / "segmentation_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    snapshot_day = int(metadata["snapshot_day"])
    lookback_days = int(metadata["lookback_days"])

    # 2. Load transactions
    engine = get_engine()

    try:
        baskets = load_basket_summary(
            engine=engine,
            start_day=snapshot_day - lookback_days,
            end_day=snapshot_day - 1,
        )
    finally:
        engine.dispose()

    # 3. Predict customer clusters
    predictions = run_segmentation_inference(baskets)

    print("\nCLUSTER PREDICTIONS COMPLETED")
    print("Clustered Customers:", len(predictions))

    print("\nCLUSTER DISTRIBUTION")
    print(predictions["cluster_name"].value_counts().to_string())

    # 4. Load RFM customers
    rfm = pd.read_csv(
        CRM_DIR / "rfm_segments.csv"
    )

    # 5. Integrate RFM and K-Means
    final_results = integrate_rfm_kmeans(
        rfm=rfm,
        clustered_customers=predictions,
    )

    # 6. Load saved training results
    original = pd.read_csv(
        SEGMENTATION_DIR / "customer_segments.csv"
    )

    compare_cols = [
        "household_key",
        "segment",
        "cluster",
        "cluster_name",
    ]

    original = (
        original[compare_cols]
        .sort_values("household_key")
        .reset_index(drop=True)
    )

    modular = (
        final_results[compare_cols]
        .sort_values("household_key")
        .reset_index(drop=True)
    )

    # 7. Compare all customer assignments
    pd.testing.assert_frame_equal(
        original,
        modular,
        check_dtype=False,
    )

    # 8. Validate customer coverage
    assert len(modular) == 2500
    assert modular["cluster"].notna().sum() == 2479
    assert modular["cluster"].isna().sum() == 21

    print("\nTotal Customers:", len(modular))
    print("Unclustered Customers:", modular["cluster"].isna().sum())

    print("\nSaved Model Inference: PASSED")
    print("Original Segmentation Comparison: PASSED")

    print("\nSEGMENTATION INFERENCE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
