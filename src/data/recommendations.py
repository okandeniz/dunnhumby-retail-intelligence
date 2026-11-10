
import pandas as pd
import numpy as np

from sqlalchemy import text
from sqlalchemy.engine import Engine


def load_basket_product_interactions(
    engine: Engine,
    end_day: int = 711,
) -> pd.DataFrame:
    """Load valid basket-product interactions from MySQL."""

    if end_day < 1:
        raise ValueError("end_day must be positive.")

    # 1. Load positive retail transactions
    query = text("""
        SELECT DISTINCT
            t.household_key,
            t.BASKET_ID,
            t.`DAY` AS basket_day,
            t.PRODUCT_ID

        FROM raw_transaction_data AS t

        INNER JOIN raw_product AS p
            ON t.PRODUCT_ID = p.PRODUCT_ID

        WHERE t.`DAY` BETWEEN 1 AND :end_day

          AND t.SALES_VALUE > 0
          AND t.QUANTITY > 0

          AND p.DEPARTMENT IS NOT NULL
          AND TRIM(p.DEPARTMENT) <> ''

          AND p.DEPARTMENT NOT IN (
              'KIOSK-GAS',
              'MISC SALES TRAN',
              'COUP/STR & MFG'
          )

          AND COALESCE(p.COMMODITY_DESC, '') NOT IN (
              'COUPON/MISC ITEMS',
              'FUEL'
          )
    """)

    with engine.connect() as connection:
        interactions = pd.read_sql_query(
            query,
            con=connection,
            params={"end_day": end_day},
        )

    # 2. Validate interactions
    if interactions.empty:
        raise ValueError("No basket-product interactions found.")

    if interactions.isna().any().any():
        raise ValueError("Missing interaction values.")

    if interactions.duplicated(
        ["BASKET_ID", "PRODUCT_ID"]
    ).any():
        raise ValueError("Duplicate basket-product pairs.")

    if (
        interactions.groupby("BASKET_ID")[
            "household_key"
        ].nunique().max() != 1
    ):
        raise ValueError("Basket has multiple households.")

    if (
        interactions.groupby("BASKET_ID")[
            "basket_day"
        ].nunique().max() != 1
    ):
        raise ValueError("Basket has multiple purchase days.")

    return interactions


def split_recommendation_data(
    interactions: pd.DataFrame,
    train_end_day: int = 621,
    test_end_day: int = 711,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into historical training and test data."""

    if not 1 <= train_end_day < test_end_day:
        raise ValueError("Invalid training or test period.")

    required = {
        "household_key",
        "BASKET_ID",
        "basket_day",
        "PRODUCT_ID",
    }

    missing = required - set(interactions.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # 1. Training period
    train_items = interactions.loc[
        interactions["basket_day"].between(
            1, train_end_day
        )
    ].copy()

    # 2. Future test period
    test_items = interactions.loc[
        interactions["basket_day"].between(
            train_end_day + 1, test_end_day
        )
    ].copy()

    # 3. Validate chronological split
    if train_items.empty or test_items.empty:
        raise ValueError("Training or test dataset is empty.")

    if train_items["basket_day"].max() > train_end_day:
        raise ValueError("Training data contains future purchases.")

    if test_items["basket_day"].min() <= train_end_day:
        raise ValueError("Test data overlaps training period.")

    if len(train_items) + len(test_items) != len(interactions):
        raise ValueError("Interactions were lost during splitting.")

    return train_items, test_items


def load_product_catalog(
    engine: Engine,
) -> pd.DataFrame:
    """Load product information from MySQL."""

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
        catalog = pd.read_sql_query(
            query,
            con=connection,
        )

    # Create initial readable product labels
    catalog["product_label"] = (
        catalog["SUB_COMMODITY_DESC"]
        .fillna("Unknown Product")
        .astype(str)
        .str.strip()
        + " | "
        + catalog["CURR_SIZE_OF_PRODUCT"]
        .fillna("")
        .astype(str)
        .str.strip()
        + " | ID: "
        + catalog["PRODUCT_ID"].astype(str)
    )

    if not catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate product IDs.")

    return catalog


def prepare_valid_catalog(
    product_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Remove non-retail products and improve product labels."""

    # 1. Define excluded departments
    excluded_departments = [
        "KIOSK-GAS",
        "MISC SALES TRAN",
        "MISC. TRANS.",
        "COUP/STR & MFG",
    ]

    # 2. Filter valid retail products
    valid_catalog = product_catalog.loc[
        ~product_catalog["DEPARTMENT"]
        .fillna("")
        .str.strip()
        .isin(excluded_departments)
    ].copy()

    # 3. Prepare product descriptions
    sub_desc = (
        valid_catalog["SUB_COMMODITY_DESC"]
        .fillna("")
        .str.strip()
    )

    commodity_desc = (
        valid_catalog["COMMODITY_DESC"]
        .fillna("")
        .str.strip()
    )

    # 4. Use commodity description as fallback
    valid_catalog["product_description"] = (
        sub_desc
        .replace({
            "": np.nan,
            "NO SUBCOMMODITY DESCRIPTION": np.nan,
        })
        .fillna(
            commodity_desc.replace({
                "": np.nan,
                "NO COMMODITY DESCRIPTION": np.nan,
            })
        )
        .fillna("Unknown Product")
    )

    # 5. Prepare product size
    product_size = (
        valid_catalog["CURR_SIZE_OF_PRODUCT"]
        .fillna("")
        .str.strip()
    )

    # 6. Build improved product labels
    valid_catalog["product_label"] = (
        valid_catalog["product_description"]
        + np.where(
            product_size.ne(""),
            " | " + product_size,
            "",
        )
        + " | ID: "
        + valid_catalog["PRODUCT_ID"].astype(str)
    )

    if not valid_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate valid product IDs.")

    return valid_catalog


def filter_retail_interactions(
    train_items: pd.DataFrame,
    test_items: pd.DataFrame,
    valid_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only products included in the valid catalog."""

    # 1. Identify valid product IDs
    valid_product_ids = set(
        valid_catalog["PRODUCT_ID"]
    )

    # 2. Filter training interactions
    train_filtered = train_items.loc[
        train_items["PRODUCT_ID"].isin(
            valid_product_ids
        )
    ].copy()

    # 3. Filter test interactions
    test_filtered = test_items.loc[
        test_items["PRODUCT_ID"].isin(
            valid_product_ids
        )
    ].copy()

    # 4. Validate filtered data
    if train_filtered.empty or test_filtered.empty:
        raise ValueError(
            "Filtered training or test data is empty."
        )

    if not train_filtered["PRODUCT_ID"].isin(
        valid_product_ids
    ).all():
        raise ValueError("Invalid training products.")

    if not test_filtered["PRODUCT_ID"].isin(
        valid_product_ids
    ).all():
        raise ValueError("Invalid test products.")

    return train_filtered, test_filtered

