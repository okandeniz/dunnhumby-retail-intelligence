
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
)


# Features selected in the original notebook
SEGMENTATION_FEATURES = [
    "purchase_frequency",
    "avg_basket_value",
    "recency",
]

# Business names from cluster profiling
CLUSTER_NAMES = {
    0: "Frequent & Recent Shoppers",
    1: "Less Active Shoppers",
}


def build_segmentation_pipeline() -> Pipeline:
    """Create the K-Means clustering pipeline."""

    preprocessing = Pipeline(
        steps=[
            (
                "log_transform",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "kmeans",
                KMeans(
                    n_clusters=2,
                    random_state=42,
                    n_init=20,
                ),
            ),
        ]
    )

    return pipeline


def fit_customer_segmentation(
    customer_features: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame]:
    """Train K-Means and assign customer clusters."""

    # 1. Validate required columns
    required = {"household_key", *SEGMENTATION_FEATURES}
    missing = required - set(customer_features.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if customer_features.empty:
        raise ValueError("Customer dataset is empty.")

    if not customer_features["household_key"].is_unique:
        raise ValueError("Duplicate customer IDs.")

    # 2. Prepare model features
    X = customer_features[SEGMENTATION_FEATURES].copy()

    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite feature values found.")

    if (values < 0).any():
        raise ValueError("Negative feature values found.")

    # 3. Train clustering pipeline
    model = build_segmentation_pipeline()

    labels = model.fit_predict(X)

    # 4. Create customer-level results
    results = customer_features.copy()

    results["cluster"] = labels

    # 5. Validate the notebook's cluster interpretation
    median_frequency = (
        results.groupby("cluster")["purchase_frequency"]
        .median()
    )

    if median_frequency.loc[0] <= median_frequency.loc[1]:
        raise ValueError(
            "Cluster interpretation changed. "
            "Review cluster labels before assigning names."
        )

    # 6. Assign business names
    results["cluster_name"] = (
        results["cluster"].map(CLUSTER_NAMES)
    )

    if results["cluster_name"].isna().any():
        raise ValueError("Unknown cluster label.")

    return model, results


def build_cluster_profile(
    segmented_customers: pd.DataFrame,
) -> pd.DataFrame:
    """Create customer behavior profiles for K-Means clusters."""

    # 1. Validate required columns
    required_cols = {
        "household_key",
        "cluster",
        "purchase_frequency",
        "avg_basket_value",
        "recency",
        "active_days",
        "total_spending",
    }

    missing = required_cols - set(segmented_customers.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if segmented_customers.empty:
        raise ValueError("Segmented customer data is empty.")

    if not segmented_customers["household_key"].is_unique:
        raise ValueError("Duplicate customer IDs.")

    # 2. Calculate cluster-level statistics
    profile = (
        segmented_customers
        .groupby("cluster")
        .agg(
            customer_count=("household_key", "size"),
            median_frequency=("purchase_frequency", "median"),
            median_basket_value=("avg_basket_value", "median"),
            median_recency=("recency", "median"),
            median_active_days=("active_days", "median"),
            avg_total_spending=("total_spending", "mean"),
            total_spending=("total_spending", "sum"),
        )
    )

    # 3. Calculate customer share
    profile["customer_share_pct"] = (
        profile["customer_count"]
        / len(segmented_customers)
        * 100
    )

    # 4. Calculate spending share
    total_spending = profile["total_spending"].sum()

    if total_spending <= 0:
        raise ValueError("Total customer spending must be positive.")

    profile["spending_share_pct"] = (
        profile["total_spending"]
        / total_spending
        * 100
    )

    # 5. Assign business cluster names
    profile["cluster_name"] = (
        profile.index.map(CLUSTER_NAMES)
    )

    if profile["cluster_name"].isna().any():
        raise ValueError("Unknown cluster labels.")

    # 6. Validate summary results
    assert profile["customer_count"].sum() == len(
        segmented_customers
    )

    assert np.isclose(
        profile["customer_share_pct"].sum(),
        100.0,
    )

    assert np.isclose(
        profile["spending_share_pct"].sum(),
        100.0,
    )

    return profile


def integrate_rfm_kmeans(
    rfm: pd.DataFrame,
    clustered_customers: pd.DataFrame,
) -> pd.DataFrame:
    """Combine RFM segments and K-Means customer clusters."""

    rfm_cols = [
        "household_key",
        "segment",
    ]

    cluster_cols = [
        "household_key",
        "cluster",
        "cluster_name",
    ]

    # 1. Validate required columns
    for data, columns in [
        (rfm, rfm_cols),
        (clustered_customers, cluster_cols),
    ]:
        missing = set(columns) - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        if data["household_key"].isna().any():
            raise ValueError("Missing customer IDs.")

        if not data["household_key"].is_unique:
            raise ValueError("Duplicate customer IDs.")

    if rfm["segment"].isna().any():
        raise ValueError("Missing RFM segments.")

    if clustered_customers[
        ["cluster", "cluster_name"]
    ].isna().any().any():
        raise ValueError("Missing K-Means cluster labels.")

    # 2. Check customer matching
    if not clustered_customers["household_key"].isin(
        rfm["household_key"]
    ).all():
        raise ValueError(
            "K-Means contains customers missing from RFM."
        )

    # 3. Keep all RFM customers
    result = rfm[rfm_cols].merge(
        clustered_customers[cluster_cols],
        on="household_key",
        how="left",
        validate="one_to_one",
    )

    # 4. Label customers without recent purchases
    result["cluster_name"] = (
        result["cluster_name"]
        .fillna("No Positive Purchases (365d)")
    )

    # 5. Validate customer coverage
    if len(result) != len(rfm):
        raise ValueError("Customer count changed.")

    if not result["household_key"].is_unique:
        raise ValueError("Duplicate customers after merge.")

    if result["cluster_name"].isna().any():
        raise ValueError("Unassigned customer groups found.")

    return result

