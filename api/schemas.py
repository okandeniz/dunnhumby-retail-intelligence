"""Public contracts for the retail dashboard."""

from typing import Annotated, Literal

from fastapi import Path, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

EntityID = Annotated[int, Path(gt=0)]
Limit = Annotated[int, Query(ge=1, le=100)]
Count = Annotated[int, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]
Status = Literal["Churn", "No Churn"]


class Features(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    purchase_frequency: Count
    avg_basket_value: Annotated[float, Field(ge=0)]
    recency: Annotated[int, Field(ge=1)]
    baskets_last_30d: Count
    baskets_prev_30d: Count
    has_purchase_365d: Annotated[int, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def check_activity(self):
        active = self.purchase_frequency > 0
        if active != bool(self.has_purchase_365d) or active != (self.recency <= 365):
            raise ValueError(
                "Yearly activity must match the purchase count and days since the last purchase."
            )
        if active != (self.avg_basket_value > 0):
            raise ValueError("Average basket value must match yearly activity.")
        if self.baskets_last_30d + self.baskets_prev_30d > self.purchase_frequency:
            raise ValueError("The two monthly counts cannot exceed the yearly count.")
        if (self.baskets_last_30d > 0) != (self.recency <= 30):
            raise ValueError(
                "Last month's count must match days since the last purchase."
            )
        if self.recency > 30 and (self.baskets_prev_30d > 0) != (self.recency <= 60):
            raise ValueError(
                "The previous month's count must match days since the last purchase."
            )
        return self


class Prediction(BaseModel):
    customer_id: int | None = None
    status: Status
    probability: Probability
    threshold: Probability
    observed: Status | None = None


class Sample(BaseModel):
    customer_id: int
    features: Features


class Product(BaseModel):
    PRODUCT_ID: int
    product_label: str


class Forecast(Product):
    DAY: int
    forecast_start_day: int
    forecast_end_day: int
    xgb_prediction: float
    baseline_last_30d: float
    baseline_avg_60d: float


class Historical(BaseModel):
    forecast_period: str
    actual: float
    xgb_prediction: float
    baseline_prediction: float
    baseline_60d_prediction: float


class ForecastMetric(BaseModel):
    model: str
    MAE: float
    RMSE: float
    wape_pct: float


class Comparison(BaseModel):
    model: str
    precision_at_k: Probability
    recall_at_k: Probability
    hit_rate_at_k: Probability
    coverage_at_k: Probability


class Recommendation(Product):
    score: float


class Ranking(BaseModel):
    score_label: str
    items: list[Recommendation]


class Segment(BaseModel):
    segment: str
    customer_count: int
    forecasted_customers: int
    avg_expected_revenue: float | None


class Cluster(BaseModel):
    cluster_name: str
    customer_count: int
    median_frequency: float
    median_recency: float
    median_basket_value: float
    spending_share_pct: float


class CrmSummary(BaseModel):
    customers: int
    clustered: int
    spending: float
    expected_revenue: float
    segments: list[Segment]
    clusters: list[Cluster]


class Customer(BaseModel):
    household_key: int
    segment: str
    cluster_name: str | None
    recency: int
    frequency: int
    monetary: float
    R_score: int
    F_score: int
    M_score: int
    expected_revenue_90d: float | None
    forecast_status: str


class Campaign(BaseModel):
    campaign_id: int
    campaign_type: str
    assignments: int
    redeemed: int
    avg_spending: float
    avg_baskets: float


class CampaignType(BaseModel):
    DESCRIPTION: str
    targeted_assignments: int
    redeemed_assignments: int
    redemption_rate_pct: float


class SpendingTest(BaseModel):
    campaign_type: str
    median_redeemed: float
    median_not_redeemed: float
    p_adjusted: Probability
    cliffs_delta: float


class RedemptionEstimate(BaseModel):
    term: str
    odds_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: Probability


class CampaignStudy(BaseModel):
    types: list[CampaignType]
    chi2: float
    p_value: Probability
    cramers_v: float
    degrees_of_freedom: int
    spending_tests: list[SpendingTest]
    redemption_estimates: list[RedemptionEstimate]
