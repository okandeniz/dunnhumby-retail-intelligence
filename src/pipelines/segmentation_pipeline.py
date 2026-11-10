
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.data.database import get_engine
from src.data.baskets import load_basket_summary
from src.features.segmentation import build_segmentation_features
from src.models.segmentation import (
    fit_customer_segmentation,
    build_cluster_profile,
    integrate_rfm_kmeans,
    SEGMENTATION_FEATURES
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "segmentation"
)

MODEL_DIR = PROJECT_ROOT / "models" / "segmentation"


def main() -> None:
    """Run the customer segmentation training pipeline."""

    print("CUSTOMER SEGMENTATION PIPELINE")
    print("-" * 40)

    # 1. Create output directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load basket-level transactions
    engine = get_engine()

    try:
        basket_df = load_basket_summary(
            engine=engine,
            start_day=347,
            end_day=711,
        )
    finally:
        engine.dispose()

    print("\nDATA LOADING COMPLETED")
    print("Total Baskets:", len(basket_df))

    # 3. Build customer features
    customer_features = build_segmentation_features(
        baskets=basket_df,
        snapshot_day=712,
        lookback_days=365,
    )

    assert customer_features["household_key"].is_unique
    assert customer_features.notna().all().all()

    print("\nFEATURE ENGINEERING COMPLETED")
    print("Customers:", len(customer_features))
    print("Features:", customer_features.shape[1] - 1)

    # 4. Train K-Means pipeline
    segmentation_model, segmented_df = (
        fit_customer_segmentation(
            customer_features=customer_features
        )
    )

    assert len(segmented_df) == len(customer_features)
    assert segmented_df["cluster"].nunique() == 2

    print("\nK-MEANS TRAINING COMPLETED")

    print("\nCLUSTER DISTRIBUTION")
    print(
        segmented_df.groupby(
            ["cluster", "cluster_name"]
        ).size().to_string()
    )

    # 5. Create cluster profiles
    cluster_profile_df = build_cluster_profile(
        segmented_customers=segmented_df
    )

    print("\nCLUSTER PROFILING COMPLETED")
    print(cluster_profile_df.round(2).to_string())

    # 6. Validate original notebook results
    assert len(segmented_df) == 2479

    print("\nSegmentation Training Validation: PASSED")

    # 7. Load RFM customer segments
    rfm_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "crm"
        / "rfm_segments.csv"
    )

    rfm_df = pd.read_csv(rfm_path)

    # 8. Integrate RFM and K-Means results
    customer_segments_df = integrate_rfm_kmeans(
        rfm=rfm_df,
        clustered_customers=segmented_df,
    )

    # 9. Validate customer coverage
    total_customers = len(customer_segments_df)

    clustered_count = (
        customer_segments_df["cluster"].notna().sum()
    )

    unclustered_count = (
        customer_segments_df["cluster"].isna().sum()
    )

    assert total_customers == 2500
    assert clustered_count == 2479
    assert unclustered_count == 21
    assert customer_segments_df["household_key"].is_unique

    print("\nRFM & K-MEANS INTEGRATION COMPLETED")
    print("Total Customers:", total_customers)
    print("Clustered Customers:", clustered_count)
    print("Unclustered Customers:", unclustered_count)

    print("\nFINAL CUSTOMER GROUP DISTRIBUTION")
    print(
        customer_segments_df["cluster_name"]
        .value_counts()
        .to_string()
    )

    # 10. Define artifact paths
    artifact_paths = {
        "features": (
            PROCESSED_DIR / "customer_features.csv"
        ),
        "segments": (
            PROCESSED_DIR / "customer_segments.csv"
        ),
        "profiles": (
            PROCESSED_DIR / "cluster_profiles.csv"
        ),
        "model": (
            MODEL_DIR / "kmeans_pipeline.joblib"
        ),
        "metadata": (
            MODEL_DIR / "segmentation_metadata.json"
        ),
    }

    # 11. Save processed datasets
    customer_features.to_csv(
        artifact_paths["features"],
        index=False,
    )

    customer_segments_df.to_csv(
        artifact_paths["segments"],
        index=False,
    )

    cluster_profile_df.reset_index().to_csv(
        artifact_paths["profiles"],
        index=False,
    )

    # 12. Save trained K-Means pipeline
    joblib.dump(
        segmentation_model,
        artifact_paths["model"],
    )

    # 13. Prepare model metadata
    metadata = {
        "model_name": "K-Means Customer Segmentation",
        "snapshot_day": 712,
        "lookback_days": 365,
        "observation_start_day": 347,
        "observation_end_day": 711,
        "n_clusters": 2,
        "random_state": 42,
        "n_init": 20,
        "selected_features": SEGMENTATION_FEATURES,
        "preprocessing": [
            "Log1p",
            "StandardScaler",
        ],
        "total_customers": int(total_customers),
        "clustered_customers": int(clustered_count),
        "unclustered_customers": int(unclustered_count),
        "cluster_names": {
            str(cluster): name
            for cluster, name in (
                segmented_df[
                    ["cluster", "cluster_name"]
                ]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
        },
    }

    # 14. Save model metadata
    with open(
        artifact_paths["metadata"],
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # 15. Reload the saved model
    loaded_model = joblib.load(
        artifact_paths["model"]
    )

    X = customer_features[SEGMENTATION_FEATURES]

    np.testing.assert_array_equal(
        segmentation_model.predict(X),
        loaded_model.predict(X),
    )

    # 16. Validate saved artifacts
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

    print("\nSEGMENTATION PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
