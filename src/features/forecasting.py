
import numpy as np
import pandas as pd


HISTORICAL_FEATURE_COLS = [
    "sales_last_7d",
    "sales_last_30d",
    "sales_prev_30d",
    "sales_last_60d",
    "rolling_std_30d",
    "active_days_last_30d",
    "baskets_last_30d",
    "sales_trend_30d",
    "days_since_last_sale",
    "week_position",
]


def build_historical_features(
    daily_product_series: pd.DataFrame,
) -> pd.DataFrame:
    """Create historical product demand features."""

    # 1. Validate required columns
    required = {
        "PRODUCT_ID",
        "DAY",
        "daily_quantity",
        "daily_baskets",
        "first_sale_day",
    }

    missing = required - set(daily_product_series.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if daily_product_series.empty:
        raise ValueError("Daily product series is empty.")

    if daily_product_series.duplicated(
        ["PRODUCT_ID", "DAY"]
    ).any():
        raise ValueError("Duplicate product-day records.")

    if daily_product_series[
        ["PRODUCT_ID", "DAY", "daily_quantity",
         "daily_baskets", "first_sale_day"]
    ].isna().any().any():
        raise ValueError("Missing values in required columns.")

    if not daily_product_series[
        ["daily_quantity", "daily_baskets"]
    ].ge(0).all().all():
        raise ValueError("Negative sales or basket values.")

    # 2. Sort historical data
    features = (
        daily_product_series
        .sort_values(["PRODUCT_ID", "DAY"])
        .reset_index(drop=True)
        .copy()
    )

    # 3. Group observations by product
    product_groups = features.groupby(
        "PRODUCT_ID",
        sort=False,
    )

    # 4. Calculate historical sales totals
    for window in [7, 30, 60]:
        features[f"sales_last_{window}d"] = (
            product_groups["daily_quantity"]
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=window,
                ).sum()
            )
        )

    # 5. Calculate previous 30-day sales
    features["sales_prev_30d"] = (
        features
        .groupby("PRODUCT_ID")["sales_last_30d"]
        .shift(30)
    )

    # 6. Calculate rolling standard deviation
    features["rolling_std_30d"] = (
        product_groups["daily_quantity"]
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=30,
            ).std()
        )
    )

    # 7. Create a temporary sales activity flag
    features["has_sale"] = (
        features["daily_quantity"] > 0
    ).astype(int)

    # 8. Count active sales days
    features["active_days_last_30d"] = (
        features
        .groupby("PRODUCT_ID")["has_sale"]
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=30,
            ).sum()
        )
    )

    # 9. Calculate recent basket activity
    features["baskets_last_30d"] = (
        product_groups["daily_baskets"]
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=30,
            ).sum()
        )
    )

    # 10. Calculate the sales trend
    features["sales_trend_30d"] = (
        features["sales_last_30d"]
        - features["sales_prev_30d"]
    )

    # 11. Find the last observed sale
    last_sale_day = features["DAY"].where(
        features["daily_quantity"].gt(0)
    )

    last_sale_day = last_sale_day.groupby(
        features["PRODUCT_ID"]
    ).ffill()

    # 12. Calculate days since last sale
    features["days_since_last_sale"] = (
        features["DAY"] - last_sale_day
    )

    # 13. Create a seven-day cycle
    features["week_position"] = (
        (features["DAY"] - 1) % 7
    ) + 1

    # 14. Calculate available product history
    features["history_days"] = (
        features["DAY"]
        - features["first_sale_day"]
        + 1
    )

    # 15. Remove temporary column
    features = features.drop(
        columns="has_sale"
    )

    # 16. Validate final dataset
    if len(features) != len(daily_product_series):
        raise ValueError("Feature engineering changed row count.")

    if not features["week_position"].between(1, 7).all():
        raise ValueError("Invalid seven-day cycle.")

    return features


def select_valid_forecast_features(
    forecast_features: pd.DataFrame,
    min_history_days: int = 60,
) -> pd.DataFrame:
    """Keep product observations with complete historical features."""

    if min_history_days < 1:
        raise ValueError("min_history_days must be positive.")

    required = {
        "PRODUCT_ID",
        "DAY",
        "history_days",
        *HISTORICAL_FEATURE_COLS,
    }

    missing = required - set(forecast_features.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # 1. Keep enough product history
    valid_features = forecast_features.loc[
        forecast_features["history_days"] >= min_history_days
    ].copy()

    # 2. Remove incomplete historical windows
    valid_features = valid_features.dropna(
        subset=HISTORICAL_FEATURE_COLS
    )

    if valid_features.empty:
        raise ValueError("No valid forecast features found.")

    # 3. Validate historical features
    if not valid_features[
        HISTORICAL_FEATURE_COLS
    ].notna().all().all():
        raise ValueError("Missing historical feature values.")

    if not np.isfinite(
        valid_features[HISTORICAL_FEATURE_COLS].to_numpy()
    ).all():
        raise ValueError("Non-finite historical features found.")

    if valid_features.duplicated(
        ["PRODUCT_ID", "DAY"]
    ).any():
        raise ValueError("Duplicate product-origin records.")

    return valid_features


def build_forecast_targets(
    daily_product_series: pd.DataFrame,
    valid_features: pd.DataFrame,
    forecast_horizon: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create future sales targets for product demand forecasting."""

    # 1. Validate parameters
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be positive.")

    required = {
        "PRODUCT_ID",
        "DAY",
        "daily_quantity",
    }

    missing = required - set(daily_product_series.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if daily_product_series.empty or valid_features.empty:
        raise ValueError("Input datasets must not be empty.")

    if daily_product_series.duplicated(
        ["PRODUCT_ID", "DAY"]
    ).any():
        raise ValueError("Duplicate product-day records.")

    if valid_features.duplicated(
        ["PRODUCT_ID", "DAY"]
    ).any():
        raise ValueError("Duplicate forecast origins.")

    # 2. Prepare daily target dataset
    forecast_targets = (
        daily_product_series[
            ["PRODUCT_ID", "DAY", "daily_quantity"]
        ]
        .sort_values(["PRODUCT_ID", "DAY"])
        .reset_index(drop=True)
        .copy()
    )

    # 3. Validate consecutive daily observations
    day_differences = (
        forecast_targets
        .groupby("PRODUCT_ID")["DAY"]
        .diff()
        .dropna()
    )

    if not day_differences.eq(1).all():
        raise ValueError("Missing days in product time series.")

    if not forecast_targets["daily_quantity"].ge(0).all():
        raise ValueError("Negative daily sales found.")

    # 4. Calculate future sales for each product
    forecast_targets["target_sales_next_30d"] = (
        forecast_targets
        .groupby("PRODUCT_ID")["daily_quantity"]
        .transform(
            lambda x: (
                x.shift(-1)
                .iloc[::-1]
                .rolling(
                    window=forecast_horizon,
                    min_periods=forecast_horizon,
                )
                .sum()
                .iloc[::-1]
            )
        )
    )

    # 5. Merge targets with historical features
    forecast_dataset = valid_features.merge(
        forecast_targets[
            [
                "PRODUCT_ID",
                "DAY",
                "target_sales_next_30d",
            ]
        ],
        on=["PRODUCT_ID", "DAY"],
        how="left",
        validate="one_to_one",
    )

    # 6. Calculate target period end
    forecast_dataset["target_end_day"] = (
        forecast_dataset["DAY"] + forecast_horizon
    )

    # 7. Keep observations with known future targets
    model_dataset = forecast_dataset.dropna(
        subset=["target_sales_next_30d"]
    ).copy()

    # 8. Validate target values
    if not model_dataset[
        "target_sales_next_30d"
    ].ge(0).all():
        raise ValueError("Negative forecast targets found.")

    max_day = daily_product_series["DAY"].max()

    if not model_dataset["target_end_day"].le(max_day).all():
        raise ValueError("Forecast target exceeds available data.")

    if len(forecast_dataset) != len(valid_features):
        raise ValueError("Forecast dataset size mismatch.")

    if model_dataset.empty:
        raise ValueError("No valid forecast targets found.")

    return (
        forecast_targets,
        forecast_dataset,
        model_dataset,
    )


def split_forecast_dataset(
    model_dataset: pd.DataFrame,
    train_end_day: int = 531,
    val_end_day: int = 621,
    test_end_day: int = 711,
    forecast_horizon: int = 30,
    first_train_origin: int = 111,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Split forecasting data into non-overlapping target periods."""

    # 1. Validate parameters
    if not (
        0 < first_train_origin
        < train_end_day
        < val_end_day
        < test_end_day
    ):
        raise ValueError("Invalid forecasting boundaries.")

    if forecast_horizon < 1:
        raise ValueError("Forecast horizon must be positive.")

    if any(
        days % forecast_horizon != 0
        for days in [
            train_end_day - first_train_origin,
            val_end_day - train_end_day,
            test_end_day - val_end_day,
        ]
    ):
        raise ValueError(
            "Forecast periods must align with the horizon."
        )

    # 2. Validate required columns
    required = {
        "PRODUCT_ID",
        "DAY",
        "target_end_day",
        "target_sales_next_30d",
    }

    missing = required - set(model_dataset.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if model_dataset.empty:
        raise ValueError("Model dataset is empty.")

    if model_dataset.duplicated(
        ["PRODUCT_ID", "DAY"]
    ).any():
        raise ValueError("Duplicate product-origin records.")

    if model_dataset[
        list(required)
    ].isna().any().any():
        raise ValueError("Missing forecast values.")

    # 3. Validate target window calculations
    if not (
        model_dataset["target_end_day"]
        == model_dataset["DAY"] + forecast_horizon
    ).all():
        raise ValueError("Invalid target end days.")

    if not model_dataset[
        "target_sales_next_30d"
    ].ge(0).all():
        raise ValueError("Negative forecast targets.")

    # 4. Define forecast origins
    train_origins = list(
        range(
            first_train_origin,
            train_end_day - forecast_horizon + 1,
            forecast_horizon,
        )
    )

    val_origins = list(
        range(
            train_end_day,
            val_end_day,
            forecast_horizon,
        )
    )

    test_origins = list(
        range(
            val_end_day,
            test_end_day,
            forecast_horizon,
        )
    )

    origin_groups = {
        "Train": train_origins,
        "Validation": val_origins,
        "Test": test_origins,
    }

    # 5. Validate non-overlapping target windows
    all_origins = (
        train_origins + val_origins + test_origins
    )

    if any(
        next_day - current_day != forecast_horizon
        for current_day, next_day in zip(
            all_origins[:-1],
            all_origins[1:],
        )
    ):
        raise ValueError("Forecast target windows overlap or have gaps.")

    # 6. Create forecasting datasets
    forecast_splits = {}

    for dataset_name, origins in origin_groups.items():

        subset = model_dataset.loc[
            model_dataset["DAY"].isin(origins)
        ].copy()

        subset["target_start_day"] = (
            subset["DAY"] + 1
        )

        subset["dataset"] = dataset_name

        if subset.empty:
            raise ValueError(
                f"{dataset_name} dataset is empty."
            )

        if not subset[
            "target_sales_next_30d"
        ].notna().all():
            raise ValueError("Missing target values.")

        if subset.duplicated(
            ["PRODUCT_ID", "DAY"]
        ).any():
            raise ValueError("Duplicate forecast origins.")

        if set(subset["DAY"]) != set(origins):
            raise ValueError(
                f"Missing origins in {dataset_name}."
            )

        forecast_splits[dataset_name] = subset

    # 7. Validate temporal boundaries
    train = forecast_splits["Train"]
    validation = forecast_splits["Validation"]
    test = forecast_splits["Test"]

    if not train["target_end_day"].le(train_end_day).all():
        raise ValueError("Training target leakage.")

    if not (
        validation["target_start_day"].gt(train_end_day).all()
        and validation["target_end_day"].le(val_end_day).all()
    ):
        raise ValueError("Validation target leakage.")

    if not (
        test["target_start_day"].gt(val_end_day).all()
        and test["target_end_day"].le(test_end_day).all()
    ):
        raise ValueError("Test target leakage.")

    # 8. Validate consistent product universe
    product_sets = [
        set(dataset["PRODUCT_ID"])
        for dataset in forecast_splits.values()
    ]

    if not all(
        products == product_sets[0]
        for products in product_sets[1:]
    ):
        raise ValueError("Product universe differs across datasets.")

    # 9. Build dataset summary
    split_summary = pd.DataFrame([
        {
            "dataset": name,
            "rows": len(dataset),
            "products": dataset["PRODUCT_ID"].nunique(),
            "forecast_origins": dataset["DAY"].nunique(),
            "first_origin": dataset["DAY"].min(),
            "last_origin": dataset["DAY"].max(),
            "target_mean": dataset[
                "target_sales_next_30d"
            ].mean(),
            "target_median": dataset[
                "target_sales_next_30d"
            ].median(),
        }
        for name, dataset in forecast_splits.items()
    ])

    return forecast_splits, split_summary


FORECAST_CATEGORICAL_COLS = [
    "PRODUCT_ID",
    "DEPARTMENT",
    "COMMODITY_DESC",
]

FORECAST_FEATURE_COLS = (
    HISTORICAL_FEATURE_COLS + FORECAST_CATEGORICAL_COLS
)


def prepare_forecast_model_inputs(
    forecast_data: pd.DataFrame,
    eligible_product_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare numeric features, categorical features and target."""

    # 1. Validate required columns
    required_data = {
        "PRODUCT_ID",
        "target_sales_next_30d",
        *HISTORICAL_FEATURE_COLS,
    }

    required_lookup = set(FORECAST_CATEGORICAL_COLS)

    if not required_data.issubset(forecast_data.columns):
        raise ValueError("Missing forecasting data columns.")

    if not required_lookup.issubset(
        eligible_product_lookup.columns
    ):
        raise ValueError("Missing product metadata columns.")

    if forecast_data.empty:
        raise ValueError("Forecast data is empty.")

    if not eligible_product_lookup["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate product metadata IDs.")

    # 2. Select product metadata
    product_metadata = eligible_product_lookup[
        FORECAST_CATEGORICAL_COLS
    ].copy()

    # 3. Add metadata to forecasting data
    model_data = forecast_data.merge(
        product_metadata,
        on="PRODUCT_ID",
        how="left",
        validate="many_to_one",
    )

    # 4. Validate product coverage
    if not model_data["PRODUCT_ID"].isin(
        product_metadata["PRODUCT_ID"]
    ).all():
        raise ValueError("Unknown products in forecast data.")

    # 5. Prepare categorical values
    model_data["PRODUCT_ID"] = (
        model_data["PRODUCT_ID"].astype(str)
    )

    for column in ["DEPARTMENT", "COMMODITY_DESC"]:
        model_data[column] = (
            model_data[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # 6. Separate features and target
    X = model_data[FORECAST_FEATURE_COLS].copy()

    y = model_data["target_sales_next_30d"].copy()

    # 7. Validate model inputs
    numeric_values = X[
        HISTORICAL_FEATURE_COLS
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Invalid numeric features.")

    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Invalid forecast targets.")

    if not y.ge(0).all():
        raise ValueError("Negative forecast targets.")

    return X, y


def prepare_forecast_inference_inputs(
    forecast_rows: pd.DataFrame,
    eligible_product_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare model features without future target values."""

    required_features = {
        "PRODUCT_ID",
        *HISTORICAL_FEATURE_COLS,
    }

    required_lookup = set(FORECAST_CATEGORICAL_COLS)

    if not required_features.issubset(forecast_rows.columns):
        raise ValueError("Missing historical forecast features.")

    if not required_lookup.issubset(
        eligible_product_lookup.columns
    ):
        raise ValueError("Missing product metadata.")

    if forecast_rows.empty:
        raise ValueError("No forecast observations found.")

    if not eligible_product_lookup["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate product IDs in lookup.")

    # 1. Add product metadata
    product_metadata = eligible_product_lookup[
        FORECAST_CATEGORICAL_COLS
    ].copy()

    model_data = forecast_rows.merge(
        product_metadata,
        on="PRODUCT_ID",
        how="left",
        validate="many_to_one",
    )

    # 2. Validate product coverage
    if not model_data["PRODUCT_ID"].isin(
        product_metadata["PRODUCT_ID"]
    ).all():
        raise ValueError("Unknown product in forecast data.")

    # 3. Prepare categorical columns
    model_data["PRODUCT_ID"] = (
        model_data["PRODUCT_ID"].astype(str)
    )

    for column in ["DEPARTMENT", "COMMODITY_DESC"]:
        model_data[column] = (
            model_data[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # 4. Select model features in training order
    X = model_data[FORECAST_FEATURE_COLS].copy()

    # 5. Validate numeric features
    numeric_values = X[
        HISTORICAL_FEATURE_COLS
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError("Invalid numeric forecast features.")

    return X
