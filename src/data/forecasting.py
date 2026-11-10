
import numpy as np
import pandas as pd

from sqlalchemy import text, bindparam
from sqlalchemy.engine import Engine


def load_forecast_product_lookup(
    engine: Engine,
) -> pd.DataFrame:
    """Load product information and create readable labels."""

    # 1. Load product catalog
    query = text("""
        SELECT
            PRODUCT_ID,
            DEPARTMENT,
            COMMODITY_DESC,
            SUB_COMMODITY_DESC,
            BRAND,
            CURR_SIZE_OF_PRODUCT
        FROM raw_product
    """)

    with engine.connect() as connection:
        product_lookup = pd.read_sql_query(
            query,
            con=connection,
        )

    # 2. Validate product IDs
    if product_lookup["PRODUCT_ID"].isna().any():
        raise ValueError("Missing product IDs.")

    if not product_lookup["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate product IDs.")

    # 3. Prepare product descriptions
    product_description = (
        product_lookup["SUB_COMMODITY_DESC"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # 4. Find missing subcategory descriptions
    missing_description = (
        product_description.eq("")
        | product_description.str.upper().str.startswith(
            "NO SUBCOMMODITY DESCRIPTION"
        )
    )

    # 5. Use commodity description as fallback
    product_description = product_description.mask(
        missing_description,
        product_lookup["COMMODITY_DESC"]
        .fillna("Unknown Product")
        .astype(str)
        .str.strip(),
    )

    # 6. Prepare product package size
    product_size = (
        product_lookup["CURR_SIZE_OF_PRODUCT"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # 7. Create readable product labels
    product_lookup["product_label"] = (
        product_description
        + np.where(
            product_size.ne(""),
            " | " + product_size,
            "",
        )
        + " | ID: "
        + product_lookup["PRODUCT_ID"].astype(str)
    )

    return product_lookup


def load_product_demand_profile(
    engine: Engine,
    product_lookup: pd.DataFrame,
    train_end_day: int = 531,
) -> pd.DataFrame:
    """Build product demand profiles from historical training data."""

    if train_end_day < 1:
        raise ValueError("train_end_day must be positive.")

    # 1. Aggregate historical product demand
    query = text("""
        SELECT
            t.PRODUCT_ID,

            COUNT(*) AS transaction_rows,

            SUM(
                CASE WHEN t.QUANTITY > 0
                THEN t.QUANTITY ELSE 0 END
            ) AS total_quantity,

            COUNT(DISTINCT
                CASE WHEN t.QUANTITY > 0
                THEN t.`DAY` ELSE NULL END
            ) AS active_sales_days,

            MIN(
                CASE WHEN t.QUANTITY > 0
                THEN t.`DAY` ELSE NULL END
            ) AS first_sale_day,

            MAX(
                CASE WHEN t.QUANTITY > 0
                THEN t.`DAY` ELSE NULL END
            ) AS last_sale_day,

            COUNT(DISTINCT
                CASE WHEN t.QUANTITY > 0
                THEN t.STORE_ID ELSE NULL END
            ) AS active_store_count,

            MAX(
                CASE WHEN t.QUANTITY > 0
                THEN t.QUANTITY ELSE NULL END
            ) AS max_transaction_qty,

            SUM(
                CASE WHEN t.QUANTITY > 100
                THEN 1 ELSE 0 END
            ) AS qty_over_100_rows,

            SUM(
                CASE WHEN t.QUANTITY <= 0
                THEN 1 ELSE 0 END
            ) AS nonpositive_qty_rows

        FROM raw_transaction_data AS t

        INNER JOIN raw_product AS p
            ON t.PRODUCT_ID = p.PRODUCT_ID

        WHERE t.`DAY` BETWEEN 1 AND :train_end_day

          AND p.DEPARTMENT NOT IN (
              'KIOSK-GAS',
              'MISC SALES TRAN'
          )

        GROUP BY t.PRODUCT_ID

        HAVING total_quantity > 0
    """)

    with engine.connect() as connection:
        profile = pd.read_sql_query(
            query,
            con=connection,
            params={"train_end_day": train_end_day},
        )

    if profile.empty:
        raise ValueError("No product demand data found.")

    # 2. Calculate observed sales history
    profile["observed_span_days"] = (
        profile["last_sale_day"]
        - profile["first_sale_day"]
        + 1
    )

    # 3. Calculate the no-sale day ratio
    profile["no_sale_day_ratio"] = (
        1
        - profile["active_sales_days"]
        / profile["observed_span_days"]
    )

    # 4. Calculate average quantity on active days
    profile["avg_qty_per_active_day"] = (
        profile["total_quantity"]
        / profile["active_sales_days"]
    )

    # 5. Calculate days since last observed sale
    profile["days_since_last_sale"] = (
        train_end_day
        - profile["last_sale_day"]
    )

    # 6. Add readable product information
    profile = profile.merge(
        product_lookup[
            ["PRODUCT_ID", "product_label", "DEPARTMENT"]
        ],
        on="PRODUCT_ID",
        how="left",
        validate="one_to_one",
    )

    # 7. Validate results
    if not profile["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate product demand profiles.")

    if profile["product_label"].isna().any():
        raise ValueError("Missing product labels.")

    if not profile["no_sale_day_ratio"].between(0, 1).all():
        raise ValueError("Invalid no-sale day ratios.")

    return profile


def select_eligible_products(
    product_demand_profile: pd.DataFrame,
    product_lookup: pd.DataFrame,
    min_active_days: int = 240,
    min_history_days: int = 365,
    max_no_sale_ratio: float = 0.50,
    max_days_since_sale: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select products with sufficient historical demand."""

    # 1. Validate parameters
    if min_active_days < 1 or min_history_days < 1:
        raise ValueError("Minimum day values must be positive.")

    if not 0 <= max_no_sale_ratio <= 1:
        raise ValueError("Invalid no-sale day ratio.")

    if max_days_since_sale < 0:
        raise ValueError("Invalid days since last sale.")

    # 2. Apply notebook selection criteria
    eligible_mask = (
        (
            product_demand_profile["active_sales_days"]
            >= min_active_days
        )
        & (
            product_demand_profile["observed_span_days"]
            >= min_history_days
        )
        & (
            product_demand_profile["no_sale_day_ratio"]
            <= max_no_sale_ratio
        )
        & (
            product_demand_profile["days_since_last_sale"]
            <= max_days_since_sale
        )
    )

    # 3. Select eligible products
    eligible_products = (
        product_demand_profile.loc[eligible_mask]
        .sort_values(
            "total_quantity",
            ascending=False,
        )
        .reset_index(drop=True)
        .copy()
    )

    if eligible_products.empty:
        raise ValueError("No eligible products found.")

    # 4. Build Streamlit product lookup
    eligible_product_lookup = product_lookup.loc[
        product_lookup["PRODUCT_ID"].isin(
            eligible_products["PRODUCT_ID"]
        )
    ].copy()

    # 5. Create category labels for Streamlit
    eligible_product_lookup["category_label"] = (
        eligible_product_lookup["DEPARTMENT"]
        .fillna("UNKNOWN")
        .str.strip()
        + " / "
        + eligible_product_lookup["COMMODITY_DESC"]
        .fillna("UNKNOWN")
        .str.strip()
    )

    # 6. Validate selected products
    if not eligible_products["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate eligible product IDs.")

    if len(eligible_product_lookup) != len(eligible_products):
        raise ValueError("Product lookup size mismatch.")

    if eligible_product_lookup[
        ["product_label", "category_label"]
    ].isna().any().any():
        raise ValueError("Missing product lookup information.")

    return eligible_products, eligible_product_lookup


def load_daily_product_sales(
    engine: Engine,
    eligible_products: pd.DataFrame,
    max_day: int = 711,
) -> pd.DataFrame:
    """Load daily sales for selected products from MySQL."""

    # 1. Validate inputs
    if max_day < 1:
        raise ValueError("max_day must be positive.")

    if "PRODUCT_ID" not in eligible_products.columns:
        raise ValueError("Missing PRODUCT_ID column.")

    if eligible_products.empty:
        raise ValueError("Eligible product dataset is empty.")

    if eligible_products["PRODUCT_ID"].isna().any():
        raise ValueError("Missing product IDs.")

    if not eligible_products["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate eligible product IDs.")

    # 2. Prepare selected product IDs
    selected_product_ids = (
        eligible_products["PRODUCT_ID"]
        .astype(int)
        .tolist()
    )

    # 3. Aggregate daily product sales
    query = text("""
        SELECT
            t.PRODUCT_ID,
            t.`DAY` AS DAY,

            SUM(t.QUANTITY) AS daily_quantity,

            COUNT(*) AS transaction_rows,

            COUNT(DISTINCT t.BASKET_ID) AS daily_baskets

        FROM raw_transaction_data AS t

        WHERE t.PRODUCT_ID IN :product_ids

          AND t.`DAY` BETWEEN 1 AND :max_day

          AND t.QUANTITY > 0

        GROUP BY
            t.PRODUCT_ID,
            t.`DAY`

        ORDER BY
            t.PRODUCT_ID,
            t.`DAY`
    """).bindparams(
        bindparam("product_ids", expanding=True)
    )

    # 4. Execute SQL query
    with engine.connect() as connection:
        daily_sales = pd.read_sql_query(
            query,
            con=connection,
            params={
                "product_ids": selected_product_ids,
                "max_day": max_day,
            },
        )

    # 5. Validate daily sales
    if daily_sales.empty:
        raise ValueError("No daily sales data found.")

    if daily_sales.duplicated(
        ["PRODUCT_ID", "DAY"]
    ).any():
        raise ValueError("Duplicate product-day records.")

    if not daily_sales["daily_quantity"].gt(0).all():
        raise ValueError("Invalid daily sales quantities.")

    if not daily_sales["DAY"].between(
        1, max_day
    ).all():
        raise ValueError("Sales outside the selected period.")

    if not daily_sales["PRODUCT_ID"].isin(
        selected_product_ids
    ).all():
        raise ValueError("Unexpected product IDs.")

    return daily_sales


def build_daily_product_series(
    daily_sales: pd.DataFrame,
    eligible_products: pd.DataFrame,
    max_day: int = 711,
) -> pd.DataFrame:
    """Create a complete daily sales panel for selected products."""

    # 1. Validate required columns
    required_sales = {
        "PRODUCT_ID",
        "DAY",
        "daily_quantity",
        "transaction_rows",
        "daily_baskets",
    }

    required_products = {
        "PRODUCT_ID",
        "first_sale_day",
    }

    missing_sales = (
        required_sales - set(daily_sales.columns)
    )

    missing_products = (
        required_products - set(eligible_products.columns)
    )

    if missing_sales or missing_products:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing_sales | missing_products)}"
        )

    if max_day < 1:
        raise ValueError("max_day must be positive.")

    if daily_sales.empty or eligible_products.empty:
        raise ValueError("Input datasets must not be empty.")

    if not eligible_products["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate eligible product IDs.")

    if daily_sales.duplicated(
        ["PRODUCT_ID", "DAY"]
    ).any():
        raise ValueError("Duplicate product-day records.")

    # 2. Get selected product IDs
    selected_product_ids = (
        eligible_products["PRODUCT_ID"]
        .astype(int)
        .tolist()
    )

    # 3. Create all product-day combinations
    full_index = pd.MultiIndex.from_product(
        [
            selected_product_ids,
            range(1, max_day + 1),
        ],
        names=["PRODUCT_ID", "DAY"],
    )

    # 4. Complete missing days with zero recorded sales
    daily_product_series = (
        daily_sales
        .set_index(["PRODUCT_ID", "DAY"])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values(["PRODUCT_ID", "DAY"])
        .reset_index(drop=True)
    )

    # 5. Get historical first sale information
    first_sales = eligible_products[
        ["PRODUCT_ID", "first_sale_day"]
    ].copy()

    # 6. Add first sale day to daily panel
    daily_product_series = daily_product_series.merge(
        first_sales,
        on="PRODUCT_ID",
        how="left",
        validate="many_to_one",
    )

    if daily_product_series["first_sale_day"].isna().any():
        raise ValueError("Missing first sale information.")

    # 7. Flag days before each product's first sale
    daily_product_series["before_first_sale"] = (
        daily_product_series["DAY"]
        < daily_product_series["first_sale_day"]
    )

    # 8. Validate complete panel
    expected_rows = len(selected_product_ids) * max_day

    if len(daily_product_series) != expected_rows:
        raise ValueError("Daily panel row count mismatch.")

    if daily_product_series.duplicated(
        ["PRODUCT_ID", "DAY"]
    ).any():
        raise ValueError("Duplicate product-day combinations.")

    if not daily_product_series[
        "daily_quantity"
    ].ge(0).all():
        raise ValueError("Negative daily sales found.")

    if (
        daily_product_series["PRODUCT_ID"].nunique()
        != len(selected_product_ids)
    ):
        raise ValueError("Missing products in daily panel.")

    return daily_product_series


def load_global_daily_activity(
    engine: Engine,
    max_day: int = 711,
) -> pd.DataFrame:
    """Load daily transaction activity across all products."""

    if max_day < 1:
        raise ValueError("max_day must be positive.")

    # 1. Query global transaction activity
    query = text("""
        SELECT
            `DAY` AS DAY,
            COUNT(*) AS all_transaction_rows,
            COUNT(DISTINCT BASKET_ID) AS all_baskets

        FROM raw_transaction_data

        WHERE `DAY` BETWEEN 1 AND :max_day

        GROUP BY `DAY`
        ORDER BY `DAY`
    """)

    # 2. Load daily activity
    with engine.connect() as connection:
        global_activity = pd.read_sql_query(
            query,
            con=connection,
            params={"max_day": max_day},
        )

    # 3. Validate source data
    if global_activity.empty:
        raise ValueError("Global activity data is empty.")

    if global_activity["DAY"].duplicated().any():
        raise ValueError("Duplicate days found.")

    if not global_activity["DAY"].between(
        1, max_day
    ).all():
        raise ValueError("Invalid activity days.")

    return global_activity


def build_daily_activity_check(
    daily_product_series: pd.DataFrame,
    global_daily_activity: pd.DataFrame,
    max_day: int = 711,
) -> pd.DataFrame:
    """Compare selected product sales with global daily activity."""

    # 1. Validate required columns
    required_panel = {
        "DAY",
        "PRODUCT_ID",
        "daily_quantity",
    }

    required_global = {
        "DAY",
        "all_transaction_rows",
        "all_baskets",
    }

    if not required_panel.issubset(
        daily_product_series.columns
    ):
        raise ValueError("Missing daily product columns.")

    if not required_global.issubset(
        global_daily_activity.columns
    ):
        raise ValueError("Missing global activity columns.")

    if max_day < 1:
        raise ValueError("max_day must be positive.")

    if daily_product_series.duplicated(
        ["PRODUCT_ID", "DAY"]
    ).any():
        raise ValueError("Duplicate product-day records.")

    if global_daily_activity["DAY"].duplicated().any():
        raise ValueError("Duplicate global activity days.")

    # 2. Aggregate selected product sales by day
    daily_total_sales = (
        daily_product_series
        .groupby("DAY", as_index=False)
        .agg(
            total_quantity=("daily_quantity", "sum"),
            active_products=(
                "daily_quantity",
                lambda x: x.gt(0).sum(),
            ),
        )
    )

    # 3. Create a complete day index
    all_days = pd.DataFrame({
        "DAY": np.arange(1, max_day + 1)
    })

    # 4. Add global activity
    daily_activity_check = all_days.merge(
        global_daily_activity,
        on="DAY",
        how="left",
        validate="one_to_one",
    )

    # 5. Fill days without recorded transactions
    daily_activity_check[
        ["all_transaction_rows", "all_baskets"]
    ] = daily_activity_check[
        ["all_transaction_rows", "all_baskets"]
    ].fillna(0)

    # 6. Add selected product sales
    daily_activity_check = daily_activity_check.merge(
        daily_total_sales,
        on="DAY",
        how="left",
        validate="one_to_one",
    )

    if daily_activity_check[
        ["total_quantity", "active_products"]
    ].isna().any().any():
        raise ValueError("Missing selected product activity.")

    # 7. Validate final dataset
    if len(daily_activity_check) != max_day:
        raise ValueError("Daily activity row count mismatch.")

    if not daily_activity_check["DAY"].is_unique:
        raise ValueError("Duplicate daily activity records.")

    if (
        daily_activity_check[
            [
                "all_transaction_rows",
                "all_baskets",
                "total_quantity",
                "active_products",
            ]
        ] < 0
    ).any().any():
        raise ValueError("Negative activity values found.")

    return daily_activity_check


def load_unusual_day_transactions(
    engine: Engine,
    unusual_days: tuple[int, ...] = (278, 643),
) -> pd.DataFrame:
    """Inspect individual transactions on unusual days."""

    if not unusual_days or any(
        day < 1 for day in unusual_days
    ):
        raise ValueError("Invalid unusual days.")

    # 1. Query unusual-day transactions
    query = text("""
        SELECT
            t.`DAY` AS DAY,
            t.STORE_ID,
            t.BASKET_ID,
            t.PRODUCT_ID,
            p.DEPARTMENT,
            p.COMMODITY_DESC,
            t.QUANTITY,
            t.SALES_VALUE

        FROM raw_transaction_data AS t

        LEFT JOIN raw_product AS p
            ON t.PRODUCT_ID = p.PRODUCT_ID

        WHERE t.`DAY` IN :unusual_days

        ORDER BY
            t.`DAY`,
            t.STORE_ID,
            t.BASKET_ID
    """).bindparams(
        bindparam("unusual_days", expanding=True)
    )

    # 2. Load transactions
    with engine.connect() as connection:
        unusual_transactions = pd.read_sql_query(
            query,
            con=connection,
            params={
                "unusual_days": list(unusual_days)
            },
        )

    # 3. Validate days
    if not unusual_transactions.empty:
        if not unusual_transactions["DAY"].isin(
            unusual_days
        ).all():
            raise ValueError("Unexpected transaction days.")

    return unusual_transactions
