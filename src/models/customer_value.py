
import numpy as np
import pandas as pd

from lifetimes import BetaGeoFitter

from src.features.cltv import build_cltv_features
from src.models.bgnbd import predict_bgnbd


def predict_customer_revenue(
    model: BetaGeoFitter,
    purchase_days: pd.DataFrame,
    snapshot_day: int = 712,
    horizon: int = 90,
) -> tuple[pd.DataFrame, int]:
    """Predict future customer purchases and revenue."""

    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")

    # 1. Build customer-level BG-NBD features
    customer_features = build_cltv_features(
        purchase_days=purchase_days,
        snapshot_day=snapshot_day,
    )

    # 2. Identify repeat customers
    repeat_mask = customer_features["frequency"].gt(0)

    # 3. Calculate population monetary mean
    population_mean = customer_features.loc[
        repeat_mask, "monetary_value"
    ].mean()

    if not np.isfinite(population_mean) or population_mean <= 0:
        raise ValueError("Invalid population monetary mean.")

    # 4. Prepare historical monetary values
    customer_features["historical_monetary"] = (
        customer_features["monetary_value"]
        .where(repeat_mask, population_mean)
    )

    # 5. Predict future purchase frequency
    predictions, corrected_count = predict_bgnbd(
        model=model,
        data=customer_features,
        horizon=horizon,
        frequency_col="frequency",
        recency_col="recency",
        age_col="T",
    )

    purchase_col = f"expected_purchases_{horizon}d"
    revenue_col = f"expected_revenue_{horizon}d"

    customer_features[purchase_col] = predictions

    # 6. Calculate future revenue
    customer_features[revenue_col] = (
        customer_features[purchase_col]
        * customer_features["historical_monetary"]
    )

    # 7. Validate predictions
    prediction_cols = [
        purchase_col,
        "historical_monetary",
        revenue_col,
    ]

    values = customer_features[
        prediction_cols
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite revenue predictions.")

    if (values < 0).any():
        raise ValueError("Negative revenue predictions.")

    if not customer_features["household_key"].is_unique:
        raise ValueError("Duplicate customers found.")

    return customer_features[
        [
            "household_key",
            "frequency",
            "recency",
            "T",
            purchase_col,
            "historical_monetary",
            revenue_col,
        ]
    ].copy(), corrected_count


def integrate_rfm_revenue(
    rfm: pd.DataFrame,
    revenue: pd.DataFrame,
    horizon: int = 90,
) -> pd.DataFrame:
    """Combine RFM segments with customer revenue forecasts."""

    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")

    revenue_col = f"expected_revenue_{horizon}d"
    purchase_col = f"expected_purchases_{horizon}d"

    rfm_cols = [
        "household_key",
        "segment",
        "R_score",
        "F_score",
        "M_score",
    ]

    revenue_cols = [
        "household_key",
        purchase_col,
        "historical_monetary",
        revenue_col,
    ]

    # 1. Validate required columns
    for data, required in [
        (rfm, rfm_cols),
        (revenue, revenue_cols),
    ]:
        missing = set(required) - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        if data["household_key"].isna().any():
            raise ValueError("Missing customer IDs.")

        if not data["household_key"].is_unique:
            raise ValueError("Duplicate customer IDs.")

    # 2. Validate forecast coverage
    if not set(revenue["household_key"]).issubset(
        set(rfm["household_key"])
    ):
        raise ValueError("Forecast contains unknown customers.")

    if revenue[revenue_cols[1:]].isna().any().any():
        raise ValueError("Incomplete revenue predictions.")

    # 3. Merge RFM and revenue predictions
    result = rfm[rfm_cols].merge(
        revenue[revenue_cols],
        on="household_key",
        how="left",
        validate="one_to_one",
    )

    # 4. Identify customers without forecasts
    result["forecast_status"] = np.where(
        result[revenue_col].notna(),
        "Forecast Available",
        "No Purchases in Last 275d",
    )

    # 5. Validate results
    assert len(result) == len(rfm)
    assert result["household_key"].is_unique

    return result


def summarize_segment_forecasts(
    crm_forecasts: pd.DataFrame,
    horizon: int = 90,
) -> pd.DataFrame:
    """Summarize expected revenue across RFM segments."""

    revenue_col = f"expected_revenue_{horizon}d"

    required = {
        "household_key",
        "segment",
        revenue_col,
    }

    missing = required - set(crm_forecasts.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # 1. Aggregate revenue by segment
    summary = (
        crm_forecasts
        .groupby("segment", as_index=False)
        .agg(
            customer_count=("household_key", "size"),
            forecasted_customers=(revenue_col, "count"),
            total_expected_revenue=(revenue_col, "sum"),
            avg_expected_revenue=(revenue_col, "mean"),
        )
    )

    # 2. Calculate forecast coverage
    summary["coverage_pct"] = (
        summary["forecasted_customers"]
        / summary["customer_count"]
        * 100
    )

    # 3. Sort by expected revenue
    return (
        summary
        .sort_values(
            "total_expected_revenue",
            ascending=False,
        )
        .reset_index(drop=True)
    )
