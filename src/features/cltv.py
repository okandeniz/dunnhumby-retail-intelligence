
import pandas as pd


def prepare_purchase_days(
    baskets: pd.DataFrame,
    snapshot_day: int = 712,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Prepare positive customer purchases by day."""

    if snapshot_day < 2 or lookback_days < 1:
        raise ValueError("Invalid observation period.")

    required_cols = {
        "household_key",
        "BASKET_ID",
        "basket_day",
        "basket_sales",
    }

    missing = required_cols - set(baskets.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = baskets.copy()

    # Keep positive purchases in the observation window
    data = data.loc[
        data["basket_sales"].gt(0)
        & data["basket_day"].between(
            snapshot_day - lookback_days,
            snapshot_day - 1,
        )
    ].copy()

    # Aggregate purchases by customer and day
    purchase_days = (
        data.groupby(
            ["household_key", "basket_day"],
            as_index=False,
        )
        .agg(
            daily_sales=("basket_sales", "sum"),
            basket_count=("BASKET_ID", "nunique"),
        )
        .sort_values(["household_key", "basket_day"])
        .reset_index(drop=True)
    )

    # Number each customer's purchase days
    purchase_days["purchase_sequence"] = (
        purchase_days
        .groupby("household_key")
        .cumcount()
        .add(1)
    )

    return purchase_days


def build_cltv_features(
    purchase_days: pd.DataFrame,
    snapshot_day: int = 712,
) -> pd.DataFrame:
    """Calculate BG-NBD and Gamma-Gamma customer features."""

    required_cols = {
        "household_key",
        "basket_day",
        "daily_sales",
        "purchase_sequence",
    }

    missing = required_cols - set(purchase_days.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if purchase_days.empty:
        raise ValueError("Purchase data is empty.")

    if purchase_days["basket_day"].ge(snapshot_day).any():
        raise ValueError("Future purchases are not allowed.")

    # 1. Calculate purchase history
    cltv = (
        purchase_days
        .groupby("household_key", as_index=False)
        .agg(
            first_purchase_day=("basket_day", "min"),
            last_purchase_day=("basket_day", "max"),
            purchase_day_count=("basket_day", "nunique"),
        )
    )

    # 2. Calculate BG-NBD features
    cltv["frequency"] = cltv["purchase_day_count"] - 1

    cltv["recency"] = (
        cltv["last_purchase_day"]
        - cltv["first_purchase_day"]
    )

    cltv["T"] = (
        snapshot_day - cltv["first_purchase_day"]
    )

    # 3. Exclude the first purchase day
    repeat_days = purchase_days.loc[
        purchase_days["purchase_sequence"] > 1
    ]

    # 4. Calculate average repeat purchase value
    repeat_monetary = (
        repeat_days
        .groupby("household_key", as_index=False)
        .agg(
            monetary_value=("daily_sales", "mean")
        )
    )

    # 5. Combine customer features
    cltv = cltv.merge(
        repeat_monetary,
        on="household_key",
        how="left",
        validate="one_to_one",
    )

    cltv["monetary_value"] = (
        cltv["monetary_value"].fillna(0)
    )

    # 6. Select model features
    cltv = cltv[
        [
            "household_key",
            "frequency",
            "recency",
            "T",
            "monetary_value",
        ]
    ]

    # 7. Validate results
    assert cltv["household_key"].is_unique
    assert cltv["frequency"].ge(0).all()
    assert cltv["recency"].le(cltv["T"]).all()
    assert cltv["monetary_value"].ge(0).all()

    return cltv


def build_calibration_holdout(
    purchase_days: pd.DataFrame,
    snapshot_day: int = 712,
    lookback_days: int = 365,
    holdout_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare calibration and holdout data for BG-NBD."""

    # 1. Validate parameters and columns
    if not (0 < holdout_days < lookback_days):
        raise ValueError("Invalid observation periods.")

    required = {"household_key", "basket_day"}
    missing = required - set(purchase_days.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if snapshot_day < lookback_days:
        raise ValueError("Invalid snapshot day.")

    data = purchase_days.copy()

    if data[["household_key", "basket_day"]].isna().any().any():
        raise ValueError("Missing customer or day values.")

    if data.duplicated(["household_key", "basket_day"]).any():
        raise ValueError("Duplicate customer-day records.")

    # 2. Define observation periods
    window_start = snapshot_day - lookback_days
    holdout_start = snapshot_day - holdout_days
    calibration_end = holdout_start - 1

    data = data.loc[
        data["basket_day"].between(
            window_start,
            snapshot_day - 1,
        )
    ].copy()

    # 3. Split purchase history
    calibration_days = data.loc[
        data["basket_day"] <= calibration_end
    ].copy()

    holdout_days_df = data.loc[
        data["basket_day"] >= holdout_start
    ].copy()

    # 4. Calculate calibration metrics
    calibration = (
        calibration_days
        .groupby("household_key", as_index=False)
        .agg(
            first_day=("basket_day", "min"),
            last_day=("basket_day", "max"),
            purchase_day_count=("basket_day", "nunique"),
        )
    )

    calibration["frequency_cal"] = (
        calibration["purchase_day_count"] - 1
    )

    calibration["recency_cal"] = (
        calibration["last_day"]
        - calibration["first_day"]
    )

    calibration["T_cal"] = (
        holdout_start - calibration["first_day"]
    )

    # 5. Count purchases during holdout
    holdout = (
        holdout_days_df
        .groupby("household_key", as_index=False)
        .agg(
            frequency_holdout=("basket_day", "nunique")
        )
    )

    # 6. Combine calibration and holdout
    result = calibration.merge(
        holdout,
        on="household_key",
        how="left",
        validate="one_to_one",
    )

    result["frequency_holdout"] = (
        result["frequency_holdout"]
        .fillna(0)
        .astype(int)
    )

    result = result[
        [
            "household_key",
            "frequency_cal",
            "recency_cal",
            "T_cal",
            "frequency_holdout",
        ]
    ]

    # 7. Validate results
    assert result["household_key"].is_unique
    assert result["frequency_cal"].ge(0).all()
    assert result["recency_cal"].ge(0).all()
    assert result["T_cal"].gt(0).all()
    assert result["recency_cal"].le(result["T_cal"]).all()

    return result, calibration_days, holdout_days_df
