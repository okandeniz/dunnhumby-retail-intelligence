
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from src.models.forecasting import BASELINE_MODELS


def evaluate_forecast(
    y_true,
    y_pred,
) -> dict[str, float]:
    """Calculate regression forecasting metrics."""

    # 1. Convert inputs to NumPy arrays
    actual = np.asarray(
        y_true,
        dtype=float,
    )

    predicted = np.asarray(
        y_pred,
        dtype=float,
    )

    # 2. Validate inputs
    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted shapes must match."
        )

    if actual.ndim != 1 or actual.size == 0:
        raise ValueError(
            "Forecast inputs must be non-empty 1D arrays."
        )

    if not np.isfinite(actual).all():
        raise ValueError("Actual values must be finite.")

    if not np.isfinite(predicted).all():
        raise ValueError("Predicted values must be finite.")

    # 3. Calculate forecast errors
    errors = predicted - actual

    # 4. Calculate evaluation metrics
    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    actual_total = np.abs(actual).sum()

    wape = (
        np.abs(errors).sum() / actual_total * 100
        if actual_total > 0
        else np.nan
    )

    bias = errors.mean()

    bias_pct = (
        errors.sum() / actual_total * 100
        if actual_total > 0
        else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE (%)": float(wape),
        "Bias": float(bias),
        "Bias (%)": float(bias_pct),
    }


def compare_baseline_models(
    baseline_data: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate and compare baseline forecasting methods."""

    required = {
        "target_sales_next_30d",
        *BASELINE_MODELS.values(),
    }

    missing = required - set(baseline_data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    results = []

    for model_name, prediction_col in BASELINE_MODELS.items():

        metrics = evaluate_forecast(
            y_true=baseline_data["target_sales_next_30d"],
            y_pred=baseline_data[prediction_col],
        )

        results.append({
            "model": model_name,
            **metrics,
        })

    comparison = (
        pd.DataFrame(results)
        .set_index("model")
        .sort_values("MAE")
    )

    return comparison


def summarize_baseline_periods(
    baseline_data: pd.DataFrame,
    forecast_horizon: int = 30,
) -> pd.DataFrame:
    """Summarize actual and predicted sales by forecast period."""

    if forecast_horizon < 1:
        raise ValueError("Forecast horizon must be positive.")

    required = {
        "DAY",
        "target_sales_next_30d",
        *BASELINE_MODELS.values(),
    }

    missing = required - set(baseline_data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # 1. Aggregate sales by forecast origin
    summary = (
        baseline_data
        .groupby("DAY")
        .agg(
            actual_total=(
                "target_sales_next_30d",
                "sum",
            ),
            last_30d_total=(
                "pred_last_30d",
                "sum",
            ),
            avg_60d_total=(
                "pred_avg_60d",
                "sum",
            ),
        )
        .reset_index()
    )

    # 2. Add forecast target end day
    summary["target_end_day"] = (
        summary["DAY"] + forecast_horizon
    )

    return summary
