from fastapi import APIRouter

from api import store
from api.schemas import Campaign, CampaignStudy, EntityID

router = APIRouter(prefix="/campaigns", tags=["Campaigns"])
DIRECTORY = store.DATA / "campaign"


@router.get("", response_model=list[int])
def campaigns():
    return sorted(
        store.csv(DIRECTORY / "campaign_spending.csv").CAMPAIGN.unique().tolist()
    )


@router.get("/study", response_model=CampaignStudy)
def study():
    """Return saved statistical tests across all household–campaign assignments."""
    return {
        **store.metadata(DIRECTORY / "chi_square_results.json"),
        "types": store.records(store.csv(DIRECTORY / "campaign_type_summary.csv")),
        "redemption_estimates": store.records(
            store.csv(DIRECTORY / "gee_redemption_summary.csv")
        ),
        "spending_tests": store.records(
            store.csv(DIRECTORY / "mann_whitney_results.csv")
        ),
    }


@router.get("/{campaign_id}", response_model=Campaign)
def campaign(campaign_id: EntityID):
    frame = store.select(
        store.csv(DIRECTORY / "campaign_spending.csv"),
        "CAMPAIGN",
        campaign_id,
        "Campaign",
    )
    return {
        "campaign_id": campaign_id,
        "campaign_type": frame.DESCRIPTION.iloc[0],
        "assignments": len(frame),
        "redeemed": frame.redeemed.sum(),
        "avg_spending": frame.campaign_spending.mean(),
        "avg_baskets": frame.campaign_baskets.mean(),
    }
