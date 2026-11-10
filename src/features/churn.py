
import pandas as pd


def build_churn_features(
    baskets: pd.DataFrame,
    snapshot_day: int,
    observation_days: int = 365,
) -> pd.DataFrame:
    """Build historical customer features without future data."""

    if observation_days < 60 or snapshot_day <= observation_days:
        raise ValueError("Invalid observation period.")

    required = {
        "household_key",
        "BASKET_ID",
        "basket_day",
        "basket_sales",
    }

    missing = required - set(baskets.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = baskets.copy()

    data["basket_sales"] = pd.to_numeric(
        data["basket_sales"],
        errors="raise",
    )

    # 1. Keep positive historical baskets
    history = data.loc[
        data["basket_sales"].gt(0)
        & data["basket_day"].lt(snapshot_day)
    ].copy()

    if history.empty:
        raise ValueError("No historical purchases found.")

    if history.duplicated(
        ["household_key", "BASKET_ID"]
    ).any():
        raise ValueError("Duplicate customer-basket pairs.")

    observation_start = snapshot_day - observation_days

    recent = history.loc[
        history["basket_day"] >= observation_start
    ]

    # 2. Identify all customers with historical purchases
    customers = history[
        ["household_key"]
    ].drop_duplicates()

    # 3. Calculate customer activity
    customer_stats = (
        recent.groupby("household_key", as_index=False)
        .agg(
            purchase_frequency=("BASKET_ID", "nunique"),
            total_spending=("basket_sales", "sum"),
            avg_basket_value=("basket_sales", "mean"),
            active_days=("basket_day", "nunique"),
        )
    )

    # 4. Find the last historical purchase
    last_purchase = (
        history.groupby("household_key", as_index=False)
        .agg(last_purchase_day=("basket_day", "max"))
    )

    # 5. Count baskets in the last 30 days
    last_30 = (
        recent.loc[
            recent["basket_day"] >= snapshot_day - 30
        ]
        .groupby("household_key", as_index=False)
        .agg(baskets_last_30d=("BASKET_ID", "nunique"))
    )

    # 6. Count baskets in the previous 30 days
    previous_30 = (
        recent.loc[
            recent["basket_day"].between(
                snapshot_day - 60,
                snapshot_day - 31,
            )
        ]
        .groupby("household_key", as_index=False)
        .agg(baskets_prev_30d=("BASKET_ID", "nunique"))
    )

    # 7. Combine historical features
    features = (
        customers
        .merge(
            customer_stats,
            on="household_key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            last_purchase,
            on="household_key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            last_30,
            on="household_key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            previous_30,
            on="household_key",
            how="left",
            validate="one_to_one",
        )
    )

    # 8. Create additional features
    features["has_purchase_365d"] = (
        features["purchase_frequency"]
        .notna()
        .astype(int)
    )

    features["recency"] = (
        snapshot_day - features["last_purchase_day"]
    )

    activity_cols = [
        "purchase_frequency",
        "total_spending",
        "avg_basket_value",
        "active_days",
        "baskets_last_30d",
        "baskets_prev_30d",
    ]

    features[activity_cols] = (
        features[activity_cols].fillna(0)
    )

    features["purchase_trend_30d"] = (
        features["baskets_last_30d"]
        - features["baskets_prev_30d"]
    )

    features = features.drop(
        columns="last_purchase_day"
    )

    if features.isna().any().any():
        raise ValueError("Missing churn feature values.")

    if not features["household_key"].is_unique:
        raise ValueError("Duplicate customers found.")

    return features


def build_churn_snapshot_dataset(
    baskets: pd.DataFrame,
    snapshot_day: int,
    observation_days: int = 365,
    target_days: int = 90,
) -> pd.DataFrame:
    """Build customer features and future churn labels."""

    if target_days < 1:
        raise ValueError("target_days must be positive.")

    target_end = snapshot_day + target_days - 1

    if target_end > baskets["basket_day"].max():
        raise ValueError("Target period exceeds available data.")

    # 1. Build features using historical data only
    features = build_churn_features(
        baskets=baskets,
        snapshot_day=snapshot_day,
        observation_days=observation_days,
    )

    # 2. Select future positive-value baskets
    future = baskets.loc[
        baskets["basket_day"].between(
            snapshot_day,
            target_end,
        )
        & baskets["basket_sales"].gt(0)
    ]

    # 3. Count future purchase days
    future_activity = (
        future.groupby("household_key", as_index=False)
        .agg(
            future_days=("basket_day", "nunique")
        )
    )

    # 4. Create binary churn target
    dataset = features.merge(
        future_activity,
        on="household_key",
        how="left",
        validate="one_to_one",
    )

    dataset["churn_90d"] = (
        dataset["future_days"]
        .fillna(0)
        .eq(0)
        .astype(int)
    )

    dataset = dataset.drop(columns="future_days")

    dataset["snapshot_day"] = snapshot_day

    # 5. Validate results
    assert dataset["household_key"].is_unique
    assert dataset.notna().all().all()
    assert dataset["churn_90d"].isin([0, 1]).all()

    return dataset
