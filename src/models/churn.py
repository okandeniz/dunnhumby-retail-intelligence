
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


NUMERIC_FEATURES = [
    "purchase_frequency",
    "avg_basket_value",
    "recency",
    "baskets_last_30d",
    "baskets_prev_30d",
]

BINARY_FEATURES = ["has_purchase_365d"]

FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES


def build_churn_pipeline() -> Pipeline:
    """Create the selected Logistic Regression pipeline."""

    # 1. Numerical preprocessing
    numeric_pipeline = Pipeline(
        steps=[
            ("log", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
        ]
    )

    # 2. Combine numerical and binary features
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ]
    )

    # 3. Create the classification pipeline
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


def fit_churn_model(
    train_df: pd.DataFrame,
) -> Pipeline:
    """Train the selected churn classification model."""

    required = set(FEATURE_COLUMNS + ["churn_90d"])
    missing = required - set(train_df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if train_df.empty:
        raise ValueError("Training data is empty.")

    # 1. Prepare model inputs
    X = train_df[FEATURE_COLUMNS].copy()
    y = train_df["churn_90d"].copy()

    # 2. Validate training data
    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Invalid feature values.")

    if (values < 0).any():
        raise ValueError("Negative feature values.")

    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Target must contain both classes: 0 and 1.")

    # 3. Train model
    model = build_churn_pipeline()
    model.fit(X, y)

    return model


def predict_churn(
    model: Pipeline,
    customer_features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Predict customer churn probabilities and classifications."""

    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    missing = set(FEATURE_COLUMNS) - set(customer_features.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if customer_features.empty:
        raise ValueError("Prediction data is empty.")

    X = customer_features[FEATURE_COLUMNS].copy()
    values = X.to_numpy(dtype=float)

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid prediction features.")

    # 1. Predict churn probabilities
    probabilities = model.predict_proba(X)[:, 1]

    # 2. Apply the selected classification threshold
    predictions = (probabilities >= threshold).astype(int)

    # 3. Prepare prediction results
    results = pd.DataFrame(index=customer_features.index)

    if "household_key" in customer_features.columns:
        results["household_key"] = customer_features["household_key"]

    results["churn_probability"] = probabilities
    results["predicted_churn"] = predictions

    results["prediction_label"] = np.where(
        predictions == 1,
        "Inactive",
        "Active",
    )

    return results.reset_index(drop=True)
