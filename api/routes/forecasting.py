from fastapi import APIRouter

from api import store
from api.schemas import EntityID, Forecast, ForecastMetric, Historical, Product

router = APIRouter(prefix="/forecasting", tags=["Forecasting"])
DIRECTORY = store.DATA / "forecasting"


@router.get("/categories")
def categories() -> list[str]:
    return sorted(
        store.csv(DIRECTORY / "eligible_product_lookup.csv").category_label.unique()
    )


@router.get("/products", response_model=list[Product])
def products(category: str):
    catalog = store.csv(DIRECTORY / "eligible_product_lookup.csv")
    return store.records(store.select(catalog, "category_label", category, "Category"))


@router.get("/evaluation", response_model=list[ForecastMetric])
def evaluation():
    frame = store.csv(DIRECTORY / "final_test_comparison.csv")
    return store.records(
        frame.rename(columns={"Unnamed: 0": "model", "WAPE (%)": "wape_pct"})
    )


@router.get("/products/{product_id}", response_model=Forecast)
def forecast(product_id: EntityID):
    """Return the saved 30-day total for each method at the final forecast origin."""
    frame = store.csv(DIRECTORY / "future_forecast.csv")
    return store.records(store.select(frame, "PRODUCT_ID", product_id, "Product"))[0]


@router.get("/products/{product_id}/history", response_model=list[Historical])
def history(product_id: EntityID):
    frame = store.csv(DIRECTORY / "test_predictions.csv")
    return store.records(
        store.select(frame, "PRODUCT_ID", product_id, "Product").sort_values("DAY")
    )
