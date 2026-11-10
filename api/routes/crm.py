from fastapi import APIRouter

from api import store
from api.schemas import CrmSummary, Customer, EntityID

router = APIRouter(prefix="/crm", tags=["Customers"])


@router.get("/summary", response_model=CrmSummary)
def summary():
    rfm = store.csv(store.DATA / "crm/rfm_segments.csv")
    clusters = store.csv(store.DATA / "segmentation/cluster_profiles.csv")
    segments = store.csv(store.DATA / "crm/crm_segment_forecast_summary.csv")
    return {
        "customers": len(rfm),
        "clustered": int(clusters.customer_count.sum()),
        "spending": rfm.monetary.sum(),
        "expected_revenue": segments.total_expected_revenue.sum(),
        "segments": store.records(segments),
        "clusters": store.records(clusters),
    }


@router.get("/customers/{customer_id}", response_model=Customer)
def customer(customer_id: EntityID):
    result = {}
    for file in [
        "crm/rfm_segments.csv",
        "segmentation/customer_segments.csv",
        "crm/crm_customer_forecasts.csv",
    ]:
        found = store.select(
            store.csv(store.DATA / file), "household_key", customer_id, "Customer"
        )
        result.update(store.records(found)[0])
    return result
