
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def load_basket_summary(
    engine: Engine,
    start_day: int | None = None,
    end_day: int | None = None,
) -> pd.DataFrame:
    """Load basket-level customer transactions from MySQL."""

    # Validate date boundaries
    if start_day is not None and start_day < 1:
        raise ValueError("start_day must be positive.")

    if end_day is not None and end_day < 1:
        raise ValueError("end_day must be positive.")

    if (
        start_day is not None
        and end_day is not None
        and start_day > end_day
    ):
        raise ValueError("start_day cannot exceed end_day.")

    # Build optional date filters
    conditions = []
    params = {}

    if start_day is not None:
        conditions.append("t.`DAY` >= :start_day")
        params["start_day"] = start_day

    if end_day is not None:
        conditions.append("t.`DAY` <= :end_day")
        params["end_day"] = end_day

    where_clause = (
        "WHERE " + " AND ".join(conditions)
        if conditions else ""
    )

    # Aggregate transactions by customer and basket
    query = text(f"""
        SELECT
            t.household_key,
            t.BASKET_ID,
            MIN(t.`DAY`) AS basket_day,
            COUNT(DISTINCT t.`DAY`) AS distinct_days,
            COUNT(*) AS basket_lines,
            COUNT(DISTINCT t.PRODUCT_ID) AS unique_products,
            SUM(t.SALES_VALUE) AS basket_sales

        FROM raw_transaction_data AS t

        {where_clause}

        GROUP BY
            t.household_key,
            t.BASKET_ID
    """)

    with engine.connect() as connection:
        basket_df = pd.read_sql_query(
            query,
            connection,
            params=params,
        )

    # Convert sales to numeric
    basket_df["basket_sales"] = pd.to_numeric(
        basket_df["basket_sales"]
    )

    # Validate basket identifiers
    if basket_df.duplicated(
        ["household_key", "BASKET_ID"]
    ).any():
        raise ValueError("Duplicate customer-basket pairs found.")

    return basket_df
