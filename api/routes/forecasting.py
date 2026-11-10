from fastapi import APIRouter

from api import store
from api.schemas import (
    EntityID,
    Forecast,
    ForecastMetric,
    Historical,
    Product,
    StoreEstimate,
)

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


@router.get("/products/{product_id}/stores", response_model=list[StoreEstimate])
def store_estimates(product_id: EntityID):
    product = forecast(product_id)
    sales = store.csv(DIRECTORY / "store_sales_180d.csv")
    recent = sales.loc[sales.PRODUCT_ID.eq(product_id) & sales.DAY.eq(product["DAY"])]
    if recent.empty:
        return []
    share = recent.units_last_180d / recent.units_last_180d.sum()
    estimates = recent.assign(
        sales_share=share, estimated_units=share * product["xgb_prediction"]
    )
    return store.records(
        estimates.sort_values(["sales_share", "STORE_ID"], ascending=[False, True])
    )
