"""Contract and artifact regression tests for the deployed application."""

import math
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import store
from api.main import app
from api.schemas import Features
from src.models.churn import FEATURE_COLUMNS
from src.pipelines.forecast_inference import run_forecast_inference
from streamlit_app.charts import historical_frame

client = TestClient(app)
DATA = store.DATA
MODELS = store.MODELS


def require(*files: Path) -> None:
    missing = [
        str(path.relative_to(store.ROOT)) for path in files if not path.is_file()
    ]
    if missing:
        pytest.skip("Missing local artifacts: " + ", ".join(missing))


def test_health_and_schema():
    assert client.get("/health").json() == {"status": "ok"}
    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["Features"]
    assert set(props["required"]) == set(FEATURE_COLUMNS)
    assert props["additionalProperties"] is False


def test_churn_prediction_and_random_fill():
    folder = DATA / "churn"
    require(
        folder / "churn_test_features.csv",
        folder / "churn_test_predictions.csv",
        MODELS / "churn/churn_pipeline.joblib",
    )
    samples = pd.read_csv(folder / "churn_test_features.csv").set_index("household_key")
    saved = pd.read_csv(folder / "churn_test_predictions.csv").set_index(
        "household_key"
    )
    for _ in range(3):
        sample = client.get("/churn/sample").json()
        cid = sample["customer_id"]
        features = sample["features"]
        assert set(features) == set(FEATURE_COLUMNS)
        for name in FEATURE_COLUMNS:
            assert features[name] == pytest.approx(samples.loc[cid, name])
        manual = client.post("/churn/predict", json=features).json()
        by_id = client.get(f"/churn/customers/{cid}").json()
        assert by_id["probability"] == pytest.approx(
            saved.loc[cid, "churn_probability"], abs=1e-10
        )
        assert by_id["probability"] == manual["probability"]
        assert by_id["threshold"] == 0.8
        assert by_id["customer_id"] == cid and manual["observed"] is None
        assert by_id["observed"] == (
            "Churn" if saved.loc[cid, "churn_90d"] else "No Churn"
        )
        assert by_id["status"] == (
            "Churn" if saved.loc[cid, "predicted_churn"] else "No Churn"
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"recency": 0},
        {"churn_90d": 1},
        {"baskets_last_30d": 15},
        {"has_purchase_365d": 0},
        {"purchase_frequency": 1.3},
    ],
)
def test_invalid_features(changes):
    values = {
        "purchase_frequency": 5,
        "avg_basket_value": 20.5,
        "recency": 7,
        "baskets_last_30d": 1,
        "baskets_prev_30d": 2,
        "has_purchase_365d": 1,
    }
    assert client.post("/churn/predict", json={**values, **changes}).status_code == 422
    assert client.post("/churn/predict", json={}).status_code == 422


def test_all_manual_rows_have_the_served_schema():
    path = DATA / "churn/churn_test_features.csv"
    require(path)
    frame = pd.read_csv(path)
    assert len(frame) == 2499 and frame.household_key.is_unique
    for row in frame[FEATURE_COLUMNS].to_dict("records"):
        Features.model_validate(row)


@pytest.mark.parametrize(
    ("url", "status"),
    [
        ("/churn/customers/-1", 422),
        ("/churn/customers/99999", 404),
        ("/crm/customers/99999", 404),
        ("/recommendations/products/10?model=NMF", 422),
        ("/recommendations/products/1?model=Item-Based%20CF", 404),
        ("/recommendations/customers/99999?model=NMF", 404),
        ("/forecasting/products/10", 404),
        ("/forecasting/products?category=Unknown", 404),
    ],
)
def test_invalid_ids(url, status):
    response = client.get(url)
    if response.status_code == 503 and "is missing" in response.json()["detail"]:
        pytest.skip("Local artifact for " + url + " is missing")
    assert response.status_code == status


def test_missing_file_is_safe(monkeypatch, tmp_path):
    from api.routes import forecasting

    monkeypatch.setattr(forecasting, "DIRECTORY", tmp_path)
    response = client.get("/forecasting/products/1082185")
    assert response.status_code == 503
    assert "missing" in response.json()["detail"]
    assert str(tmp_path) not in response.text


@pytest.mark.parametrize(
    ("model", "first"),
    [
        ("Truncated SVD", 981760),
        ("NMF", 1027569),
        ("User-Based CF", 1127831),
        ("Association Rules", 8090537),
        ("Popularity Baseline", None),
        ("Item-Based CF", 1127831),
        ("Content-Based Filtering", None),
    ],
)
def test_customer_recommendations(model, first):
    response = client.get(
        "/recommendations/customers/1", params={"model": model, "top_n": 5}
    )
    if response.status_code == 503 and "is missing" in response.json()["detail"]:
        pytest.skip(f"Missing {model} artifact")
    assert response.status_code == 200
    rows = response.json()["items"]
    assert len(rows) == 5 and response.json()["score_label"]
    ids = [r["PRODUCT_ID"] for r in rows]
    assert len(set(ids)) == 5 and all(
        r["product_label"] and math.isfinite(r["score"]) for r in rows
    )
    if first:
        assert ids[0] == first
    purchased = store.artifact(
        MODELS / "recommendation/association_rules/train_purchase_sets.joblib"
    )
    assert not set(ids).intersection(purchased[1])


@pytest.mark.parametrize(
    "model", ["Association Rules", "Item-Based CF", "Content-Based Filtering"]
)
def test_product_recommendations(model):
    response = client.get(
        "/recommendations/products/826249", params={"model": model, "top_n": 3}
    )
    if response.status_code == 503 and "is missing" in response.json()["detail"]:
        pytest.skip(f"Missing {model} artifact")
    assert response.status_code == 200
    ids = [r["PRODUCT_ID"] for r in response.json()["items"]]
    assert 0 < len(ids) <= 3 and 826249 not in ids
    if model == "Association Rules":
        assert ids == [1098066, 1082185, 1029743]


def test_crm_and_campaign():
    require(DATA / "crm/rfm_segments.csv", DATA / "campaign/campaign_spending.csv")
    summary = client.get("/crm/summary").json()
    assert summary["customers"] == 2500
    assert sum(c["customer_count"] for c in summary["clusters"]) == 2479
    assert client.get("/crm/customers/102").json()["expected_revenue_90d"] is None
    study = client.get("/campaigns/study").json()
    assert sum(t["targeted_assignments"] for t in study["types"]) == 7208
    assert study["p_value"] == pytest.approx(3.549996529166533e-24, abs=1e-35)
    assert client.get("/campaigns/8").json()["campaign_type"] == "TypeA"


def test_forecast_api_and_chart_data():
    folder = DATA / "forecasting"
    require(folder / "future_forecast.csv", folder / "test_predictions.csv")
    row = client.get("/forecasting/products/1082185").json()
    assert (row["DAY"], row["forecast_start_day"], row["forecast_end_day"]) == (
        711,
        712,
        741,
    )
    assert row["xgb_prediction"] == pytest.approx(1241.66, abs=0.02)
    assert (row["baseline_last_30d"], row["baseline_avg_60d"]) == (1565, 1554)
    history = client.get("/forecasting/products/1082185/history").json()
    frame = historical_frame(history)
    assert len(frame) == 12 and frame.Method.nunique() == 4
    assert frame.forecast_period.unique().tolist() == ["622-651", "652-681", "682-711"]
    assert len(client.get("/forecasting/evaluation").json()) == 3


def test_saved_xgboost_inference():
    require(
        MODELS / "forecasting/xgb_log_target.joblib",
        DATA / "forecasting/daily_product_series.csv",
        DATA / "forecasting/eligible_product_lookup.csv",
        MODELS / "forecasting/metadata.json",
    )
    result = run_forecast_inference(product_id=1082185)
    assert result.xgb_prediction.item() == pytest.approx(1241.66, abs=0.02)
