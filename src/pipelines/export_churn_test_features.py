"""Export the historical test inputs and check every score against the saved run."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd

from src.features.churn import build_churn_features
from src.models.churn import FEATURE_COLUMNS, predict_churn

ROOT = Path(__file__).resolve().parents[2]


def export(root: Path = ROOT) -> Path:
    """Recreate day 622 from local retail transactions, without fitting a model."""
    model_dir = root / "models/churn"
    out_dir = root / "data/processed/churn"
    metadata = json.loads((model_dir / "churn_model_metadata.json").read_text())
    day = metadata["test_snapshot"]
    columns = ["household_key", "BASKET_ID", "DAY", "SALES_VALUE"]
    rows = pd.read_csv(root / "data/raw/transaction_data.csv", usecols=columns)
    history = rows.loc[rows.DAY < day]
    # Sum money as fixed-point integers, as in the SQL basket query.
    baskets = (
        history.assign(amount=(history.SALES_VALUE * 10000).round().astype("int64"))
        .groupby(["household_key", "BASKET_ID"], as_index=False)
        .agg(basket_day=("DAY", "min"), basket_sales=("amount", "sum"))
    )
    baskets["basket_sales"] /= 10000
    features = build_churn_features(baskets, day, metadata["observation_days"])
    features = features[["household_key", *FEATURE_COLUMNS]].sort_values(
        "household_key"
    )
    counts = features.columns.difference(["avg_basket_value"])
    features = features.astype(dict.fromkeys(counts, "int64"))
    model = joblib.load(model_dir / "churn_pipeline.joblib")
    result = predict_churn(model, features, metadata["threshold"])
    saved = pd.read_csv(out_dir / "churn_test_predictions.csv").sort_values(
        "household_key"
    )
    np.testing.assert_array_equal(result.household_key, saved.household_key)
    np.testing.assert_allclose(
        result.churn_probability, saved.churn_probability, atol=1e-10, rtol=0
    )
    np.testing.assert_array_equal(result.predicted_churn, saved.predicted_churn)
    path = out_dir / "churn_test_features.csv"
    with NamedTemporaryFile(
        dir=out_dir, suffix=".tmp", delete=False, mode="w", newline=""
    ) as tmp:
        features.to_csv(tmp, index=False)
    Path(tmp.name).replace(path)
    return path


if __name__ == "__main__":
    print(export())
