from typing import Annotated

from fastapi import APIRouter, Query

from api import store
from api.schemas import Comparison, EntityID, Limit, Product, Ranking
from api.services import recommendations as service

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])
TopN = Annotated[int, Query(ge=1, le=50)]


@router.get("/models")
def models() -> dict[str, list[str]]:
    return {
        "Customer": list(service.CUSTOMER_SCORES),
        "Product": list(service.PRODUCT_SCORES),
    }


@router.get("/comparison", response_model=list[Comparison])
def comparison():
    return store.records(
        store.csv(store.DATA / "recommendation/model_comparison_stage4.csv")
    )


@router.get("/products", response_model=list[Product])
def products(
    model: str,
    q: Annotated[str, Query(max_length=100)] = "",
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Limit = 50,
):
    frame = service.catalog(model)
    matches = frame.loc[
        frame.product_label.str.contains(q.strip(), case=False, regex=False, na=False)
    ]
    return store.records(matches.iloc[offset : offset + limit])


@router.get("/products/{product_id}", response_model=Ranking)
def product(product_id: EntityID, model: str, top_n: TopN = 10):
    result = service.product_ranking(model, product_id, top_n)
    return {
        "score_label": service.PRODUCT_SCORES[model],
        "items": store.records(result),
    }


@router.get("/customers/{customer_id}", response_model=Ranking)
def customer(customer_id: EntityID, model: str, top_n: TopN = 10):
    result = service.customer_ranking(model, customer_id, top_n)
    return {
        "score_label": service.CUSTOMER_SCORES[model],
        "items": store.records(result),
    }
