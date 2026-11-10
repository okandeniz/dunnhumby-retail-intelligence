
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


from xgboost import XGBRegressor

from src.features.forecasting import (
    HISTORICAL_FEATURE_COLS,
    FORECAST_CATEGORICAL_COLS,
)



BASELINE_MODELS = {
    "Last 30 Days": "pred_last_30d",
    "Last 60-Day Average": "pred_avg_60d",
}


def add_baseline_predictions(
    forecast_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create baseline forecasts using historical sales."""

    # 1. Validate required columns
    required = {
        "sales_last_30d",
        "sales_prev_30d",
        "sales_last_60d",
    }

    missing = required - set(forecast_data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if forecast_data.empty:
        raise ValueError("Forecast dataset is empty.")

    # 2. Create a copy
    predictions = forecast_data.copy()

    historical_sales = predictions[
        list(required)
    ].to_numpy(dtype=float)

    if not np.isfinite(historical_sales).all():
        raise ValueError("Invalid historical sales values.")

    if (historical_sales < 0).any():
        raise ValueError("Negative historical sales found.")

    # 3. Last 30 Days Baseline
    predictions["pred_last_30d"] = (
        predictions["sales_last_30d"]
    )

    # 4. Last 60-Day Average Baseline
    predictions["pred_avg_60d"] = (
        predictions["sales_last_60d"] / 2
    )

    # 5. Validate historical sales consistency
    if not np.allclose(
        predictions["sales_last_60d"],
        predictions["sales_last_30d"]
        + predictions["sales_prev_30d"],
    ):
        raise ValueError(
            "Historical 60-day sales do not match "
            "the two 30-day periods."
        )

    # 6. Validate predictions
    prediction_columns = list(
        BASELINE_MODELS.values()
    )

    if not predictions[
        prediction_columns
    ].notna().all().all():
        raise ValueError("Missing baseline predictions.")

    if not predictions[
        prediction_columns
    ].ge(0).all().all():
        raise ValueError("Negative baseline predictions.")

    return predictions

def build_xgb_forecast_pipeline() -> Pipeline:
    """Create the baseline XGBoost forecasting pipeline."""

    # 1. Define preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                HISTORICAL_FEATURE_COLS,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                FORECAST_CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )

    # 2. Initialize XGBoost
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mae",
        random_state=42,
        n_jobs=4,
    )

    # 3. Create model pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def predict_forecast_demand(
    pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict non-negative product demand."""

    # 1. Generate predictions
    predictions = pipeline.predict(X)

    # 2. Prevent negative sales predictions
    predictions = np.clip(
        predictions,
        a_min=0,
        a_max=None,
    )

    # 3. Validate predictions
    if len(predictions) != len(X):
        raise ValueError("Prediction count mismatch.")

    if not np.isfinite(predictions).all():
        raise ValueError("Invalid forecast predictions.")

    return predictions


def build_selected_forecast_model() -> TransformedTargetRegressor:
    """Create the selected XGBoost model with log target."""

    return TransformedTargetRegressor(
        regressor=build_xgb_forecast_pipeline(),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def predict_selected_forecast_model(
    model: TransformedTargetRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict non-negative demand on the original sales scale."""

    predictions = np.asarray(
        model.predict(X),
        dtype=float,
    )

    predictions = np.clip(
        predictions,
        a_min=0,
        a_max=None,
    )

    if len(predictions) != len(X):
        raise ValueError("Prediction count mismatch.")

    if not np.isfinite(predictions).all():
        raise ValueError("Invalid forecast predictions.")

    return predictions
