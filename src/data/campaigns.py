
import pandas as pd

from sqlalchemy import text
from sqlalchemy.engine import Engine


def load_campaign_spending(
    engine: Engine,
    participation: pd.DataFrame,
) -> pd.DataFrame:
    """Load campaign-period spending for targeted households."""

    # 1. Validate participation data
    required = {
        "CAMPAIGN",
        "household_key",
        "DESCRIPTION",
        "redeemed",
    }

    missing = required - set(participation.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if participation.empty:
        raise ValueError("Campaign participation data is empty.")

    if participation.duplicated(
        ["CAMPAIGN", "household_key"]
    ).any():
        raise ValueError("Duplicate campaign assignments.")

    # 2. Calculate spending within campaign dates
    query = text("""
        WITH basket_totals AS (
            SELECT
                household_key,
                BASKET_ID,
                MIN(`DAY`) AS basket_day,
                SUM(SALES_VALUE) AS basket_sales
            FROM raw_transaction_data
            GROUP BY household_key, BASKET_ID
            HAVING SUM(SALES_VALUE) > 0
        )
        SELECT
            ct.CAMPAIGN,
            ct.household_key,
            COUNT(bt.BASKET_ID) AS campaign_baskets,
            COALESCE(SUM(bt.basket_sales), 0) AS campaign_spending
        FROM (
            SELECT DISTINCT CAMPAIGN, household_key
            FROM raw_campaign_table
        ) AS ct
        JOIN raw_campaign_desc AS cd
            ON ct.CAMPAIGN = cd.CAMPAIGN
        LEFT JOIN basket_totals AS bt
            ON ct.household_key = bt.household_key
            AND bt.basket_day BETWEEN cd.START_DAY AND cd.END_DAY
        GROUP BY ct.CAMPAIGN, ct.household_key
    """)

    # 3. Load campaign sales
    with engine.connect() as connection:
        campaign_sales = pd.read_sql_query(
            query,
            con=connection,
        )

    campaign_sales["campaign_spending"] = pd.to_numeric(
        campaign_sales["campaign_spending"],
        errors="raise",
    )

    # 4. Merge sales with redemption records
    result = participation.merge(
        campaign_sales,
        on=["CAMPAIGN", "household_key"],
        how="left",
        validate="one_to_one",
    )

    # 5. Validate results
    if len(result) != len(participation):
        raise ValueError("Campaign assignment count changed.")

    if result[
        ["campaign_baskets", "campaign_spending"]
    ].isna().any().any():
        raise ValueError("Missing campaign spending values.")

    if result["campaign_spending"].lt(0).any():
        raise ValueError("Negative campaign spending found.")

    if result["campaign_baskets"].lt(0).any():
        raise ValueError("Negative basket counts found.")

    return result
