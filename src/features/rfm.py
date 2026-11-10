
import numpy as np
import pandas as pd


def build_rfm_features(
    baskets: pd.DataFrame,
    snapshot_day: int = 712,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Calculate RFM metrics from positive-value shopping baskets."""

    if snapshot_day < 2 or lookback_days < 1:
        raise ValueError("Invalid snapshot day or lookback period.")

    required_cols = {
        "household_key",
        "BASKET_ID",
        "basket_day",
        "basket_sales",
    }

    missing_cols = required_cols - set(baskets.columns)

    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    data = baskets[list(required_cols)].copy()

    data["basket_sales"] = pd.to_numeric(
        data["basket_sales"],
        errors="raise",
    )

    # Keep only positive purchases before the snapshot
    positive_baskets = data.loc[
        data["basket_sales"].gt(0)
        & data["basket_day"].lt(snapshot_day)
    ].copy()

    if positive_baskets.empty:
        raise ValueError("No positive baskets before snapshot.")

    if positive_baskets.duplicated(
        ["household_key", "BASKET_ID"]
    ).any():
        raise ValueError("Duplicate customer-basket pairs.")

    # Include all customers with previous positive purchases
    customers = positive_baskets[
        ["household_key"]
    ].drop_duplicates()

    # Find the last purchase day
    last_purchases = (
        positive_baskets
        .groupby("household_key", as_index=False)
        .agg(last_purchase_day=("basket_day", "max"))
    )

    # Select observation period
    window_baskets = positive_baskets.loc[
        positive_baskets["basket_day"].between(
            snapshot_day - lookback_days,
            snapshot_day - 1,
        )
    ]

    # Calculate frequency and monetary value
    window_metrics = (
        window_baskets
        .groupby("household_key", as_index=False)
        .agg(
            frequency=("BASKET_ID", "nunique"),
            monetary=("basket_sales", "sum"),
        )
    )

    # Combine customer metrics
    rfm = (
        customers
        .merge(
            last_purchases,
            on="household_key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            window_metrics,
            on="household_key",
            how="left",
            validate="one_to_one",
        )
    )

    rfm["recency"] = snapshot_day - rfm["last_purchase_day"]
    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["monetary"] = rfm["monetary"].fillna(0)

    return rfm[
        ["household_key", "recency", "frequency", "monetary"]
    ]


def calculate_rfm_score(
    series: pd.Series,
    higher_is_better: bool = True,
    n_groups: int = 5,
) -> pd.Series:
    """Assign RFM scores using percentile ranks."""

    if n_groups < 1:
        raise ValueError("n_groups must be positive.")

    percentile_ranks = series.rank(
        method="average",
        pct=True,
        ascending=higher_is_better,
    )

    return (
        np.ceil(percentile_ranks * n_groups)
        .clip(lower=1, upper=n_groups)
        .astype(int)
    )


def add_rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Calculate R, F, M scores and the combined RFM score."""

    result = rfm.copy()

    result["R_score"] = calculate_rfm_score(
        result["recency"],
        higher_is_better=False,
    )

    result["F_score"] = calculate_rfm_score(
        result["frequency"],
        higher_is_better=True,
    )

    result["M_score"] = calculate_rfm_score(
        result["monetary"],
        higher_is_better=True,
    )

    result["RFM_score"] = (
        result[["R_score", "F_score", "M_score"]]
        .astype(str)
        .agg("".join, axis=1)
    )

    return result


def assign_rfm_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign the final eight RFM customer segments."""

    result = rfm.copy()

    r = result["R_score"]
    f = result["F_score"]
    m = result["M_score"]

    conditions = [
        result["frequency"].eq(0),
        r.ge(4) & f.ge(4) & m.ge(4),
        r.ge(4) & f.ge(4),
        r.ge(4) & f.between(2, 3),
        r.ge(4) & f.eq(1),
        r.eq(3),
        r.le(2) & f.ge(3),
        r.le(2) & f.le(2),
    ]

    segment_names = [
        "Dormant (365d)",
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "Recent Low Frequency",
        "Recently Active",
        "Less Recent - Frequent",
        "Less Recent - Infrequent",
    ]

    result["segment"] = np.select(
        conditions,
        segment_names,
        default="Unassigned",
    )

    if result["segment"].eq("Unassigned").any():
        raise ValueError("Some customers have no RFM segment.")

    return result
