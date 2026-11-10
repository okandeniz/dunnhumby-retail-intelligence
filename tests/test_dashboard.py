"""Exercise the dashboard widgets against the same FastAPI application."""

from urllib.parse import urlsplit

import pytest
import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api import store
from api.main import app


def test_dashboard_and_editable_random_fill(monkeypatch):
    from streamlit import cache_data

    client = TestClient(app)
    cache_data.clear()

    def send(method, url, **kwargs):
        return client.request(
            method, urlsplit(url).path, params=kwargs["params"], json=kwargs["json"]
        )

    monkeypatch.setattr(requests, "request", send)
    if not (store.DATA / "churn/churn_test_features.csv").is_file():
        pytest.skip("Missing churn test inputs")
    at = AppTest.from_file(
        str(store.ROOT / "streamlit_app/app.py"), default_timeout=40
    ).run()
    assert not at.exception and not at.warning
    forecast_page = next(
        tab for tab in at.tabs if tab.label == "Product Demand Forecasting"
    )
    assert [tab.label for tab in forecast_page.tabs] == [
        "Next 30-day forecast",
        "Actual and predicted demand",
        "Forecast by store",
    ]
    assert all(len(tab.get("plotly_chart")) == 1 for tab in forecast_page.tabs)
    category = next(s for s in at.selectbox if s.label == "Category")
    category.select("GROCERY / CHEESE").run()
    available = client.get(
        "/forecasting/products", params={"category": "GROCERY / CHEESE"}
    ).json()
    assert at.selectbox(key="forecast_product").options == [
        p["product_label"] for p in available
    ]
    assert not at.exception and not at.warning

    def button(label):
        return next(b for b in at.button if b.label == label)

    for _ in range(2):
        button("Random Fill").click().run()
        assert not at.exception
        assert at.number_input(key="manual_recency").value > 0
        button("Predict Churn").click().run()
        assert not at.exception and not at.warning
        assert "Churn probability" in [m.label for m in at.metric]
    at.number_input(key="manual_avg_basket_value").set_value(0.0)
    at.number_input(key="manual_purchase_frequency").set_value(10)
    button("Predict Churn").click().run()
    assert at.warning and not at.exception
    cache_data.clear()
