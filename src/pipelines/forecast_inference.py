
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.forecasting import (
    build_historical_features,
    select_valid_forecast_features,
    prepare_forecast_inference_inputs,
)

from src.models.forecasting import (
    predict_selected_forecast_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "forecasting"

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed" / "forecasting"
)


def run_forecast_inference(
    product_id: int | None = None,
) -> pd.DataFrame:
    """Predict future product demand using saved model artifacts."""

    # 1. Load saved model
    model = joblib.load(
        MODEL_DIR / "xgb_log_target.joblib"
    )

    # 2. Load model metadata
    with open(
        MODEL_DIR / "metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    # 3. Load saved product lookup and daily history
    product_lookup = pd.read_csv(
        DATA_DIR / "eligible_product_lookup.csv"
    )

    daily_series = pd.read_csv(
        DATA_DIR / "daily_product_series.csv"
    )

    # 4. Validate inference configuration
    last_day = int(metadata["last_observed_day"])
    horizon = int(metadata["forecast_horizon"])

    if horizon != 30:
        raise ValueError("Unexpected forecasting horizon.")

    if daily_series["DAY"].max() != last_day:
        raise ValueError("Daily history does not match metadata.")

    if len(product_lookup) != metadata["products"]:
        raise ValueError("Product count mismatch.")

    # 5. Recreate historical features
    historical_features = build_historical_features(
        daily_product_series=daily_series,
    )

    valid_features = select_valid_forecast_features(
        forecast_features=historical_features,
        min_history_days=60,
    )

    # 6. Select the latest observed day
    latest_features = (
        valid_features.loc[
            valid_features["DAY"].eq(last_day)
        ]
        .sort_values("PRODUCT_ID")
        .reset_index(drop=True)
        .copy()
    )

    if len(latest_features) != metadata["products"]:
        raise ValueError("Missing eligible products on last day.")

    if not latest_features["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate products at forecast origin.")

    # 7. Prepare model inputs
    X_future = prepare_forecast_inference_inputs(
        forecast_rows=latest_features,
        eligible_product_lookup=product_lookup,
    )

    if X_future.columns.tolist() != metadata["feature_names"]:
        raise ValueError("Model feature schema mismatch.")

    # 8. Predict demand using saved model
    predictions = predict_selected_forecast_model(
        model=model,
        X=X_future,
    )

    # 9. Prepare output dataset
    results = latest_features[
        [
            "PRODUCT_ID",
            "DAY",
            "sales_last_30d",
            "sales_last_60d",
        ]
    ].copy()

    results["forecast_start_day"] = last_day + 1
    results["forecast_end_day"] = last_day + horizon

    results["xgb_prediction"] = predictions

    # 10. Add baseline forecasts
    results["baseline_last_30d"] = (
        results["sales_last_30d"]
    )

    results["baseline_avg_60d"] = (
        results["sales_last_60d"] / 2
    )

    # 11. Add readable product information
    results = results.merge(
        product_lookup[
            [
                "PRODUCT_ID",
                "product_label",
                "DEPARTMENT",
                "COMMODITY_DESC",
            ]
        ],
        on="PRODUCT_ID",
        how="left",
        validate="one_to_one",
    )

    results["forecast_period"] = (
        results["forecast_start_day"].astype(str)
        + "-"
        + results["forecast_end_day"].astype(str)
    )

    results["model_version"] = metadata["model_version"]

    # 12. Validate predictions
    prediction_columns = [
        "xgb_prediction",
        "baseline_last_30d",
        "baseline_avg_60d",
    ]

    if not np.isfinite(
        results[prediction_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid forecast predictions.")

    if not results[prediction_columns].ge(0).all().all():
        raise ValueError("Negative forecast predictions.")

    if results["product_label"].isna().any():
        raise ValueError("Missing product descriptions.")

    # 13. Filter a selected product if requested
    if product_id is not None:
        if product_id not in results["PRODUCT_ID"].values:
            raise ValueError(
                f"Product {product_id} is not available."
            )

        results = results.loc[
            results["PRODUCT_ID"].eq(product_id)
        ].copy()

    # 14. Sort by predicted demand
    results = (
        results
        .sort_values(
            ["xgb_prediction", "PRODUCT_ID"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    return results


def main() -> None:
    """Run saved model inference from the terminal."""

    parser = argparse.ArgumentParser(
        description="Product Demand Forecasting Inference"
    )

    parser.add_argument(
        "--product-id",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if args.top_n < 1:
        raise ValueError("top-n must be positive.")

    print("PRODUCT DEMAND FORECASTING INFERENCE")
    print("-" * 45)

    # 1. Generate predictions
    results = run_forecast_inference(
        product_id=args.product_id,
    )

    print("\nSAVED MODEL INFERENCE: PASSED")
    print("Forecast Origin:", results["DAY"].iloc[0])
    print("Forecast Period:", results["forecast_period"].iloc[0])
    print("Available Products:", len(results))

    # 2. Display predictions
    display_columns = [
        "PRODUCT_ID",
        "product_label",
        "xgb_prediction",
        "baseline_last_30d",
        "baseline_avg_60d",
    ]

    print("\nFORECAST RESULTS")

    print(
        results[display_columns]
        .head(args.top_n)
        .round(2)
        .to_string(index=False)
    )

    # 3. Save all product predictions
    if args.product_id is None:
        output_path = DATA_DIR / "future_forecast.csv"

        results.to_csv(
            output_path,
            index=False,
        )

        print("\nForecast Dataset Saved:", output_path)

    print(
        "\nFORECAST INFERENCE PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
