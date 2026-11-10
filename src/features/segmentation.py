
import pandas as pd


def build_segmentation_features(
    baskets: pd.DataFrame,
    snapshot_day: int = 712,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Build customer features for K-Means segmentation."""

    if snapshot_day < 2 or lookback_days < 1:
        raise ValueError("Invalid observation period.")

    # 1. Validate required columns
    required_cols = {
        "household_key",
        "BASKET_ID",
        "basket_day",
        "basket_lines",
        "unique_products",
        "basket_sales",
    }

    missing = required_cols - set(baskets.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = baskets.copy()

    data["basket_sales"] = pd.to_numeric(
        data["basket_sales"],
        errors="raise",
    )

    # 2. Select positive-value baskets in the observation period
    start_day = snapshot_day - lookback_days

    data = data.loc[
        data["basket_day"].between(
            start_day,
            snapshot_day - 1,
        )
        & data["basket_sales"].gt(0)
    ].copy()

    if data.empty:
        raise ValueError("No eligible baskets found.")

    if data.duplicated(["household_key", "BASKET_ID"]).any():
        raise ValueError("Duplicate customer-basket pairs found.")

    # 3. Calculate customer-level features
    customer_features = (
        data
        .groupby("household_key", as_index=False)
        .agg(
            purchase_frequency=("BASKET_ID", "nunique"),
            total_spending=("basket_sales", "sum"),
            avg_basket_value=("basket_sales", "mean"),
            active_days=("basket_day", "nunique"),
            avg_basket_lines=("basket_lines", "mean"),
            avg_unique_products_per_basket=(
                "unique_products", "mean"
            ),
            last_purchase_day=("basket_day", "max"),
        )
    )

    # 4. Calculate purchase recency
    customer_features["recency"] = (
        snapshot_day - customer_features["last_purchase_day"]
    )

    customer_features = customer_features.drop(
        columns="last_purchase_day"
    )

    # 5. Validate output
    if not customer_features["household_key"].is_unique:
        raise ValueError("Duplicate customers found.")

    if customer_features.isna().any().any():
        raise ValueError("Missing customer feature values.")

    return customer_features
