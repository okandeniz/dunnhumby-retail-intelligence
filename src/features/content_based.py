
import pandas as pd


CONTENT_FEATURES = [
    "DEPARTMENT",
    "COMMODITY_DESC",
    "SUB_COMMODITY_DESC",
]


def prepare_content_catalog(
    valid_catalog: pd.DataFrame,
    product_ids: pd.Index,
) -> pd.DataFrame:
    """Prepare product text features for Content-Based Filtering."""

    # 1. Validate required columns
    required = {
        "PRODUCT_ID",
        "product_label",
        *CONTENT_FEATURES,
    }

    missing = required - set(valid_catalog.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if not valid_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate product IDs.")

    if not product_ids.is_unique or len(product_ids) == 0:
        raise ValueError("Invalid product index.")

    # 2. Check that all model products exist
    catalog_ids = pd.Index(valid_catalog["PRODUCT_ID"])

    missing_products = product_ids.difference(catalog_ids)

    if len(missing_products) > 0:
        raise ValueError(
            f"Products missing from catalog: {len(missing_products)}"
        )

    # 3. Select products used by Collaborative Filtering
    content_catalog = (
        valid_catalog
        .set_index("PRODUCT_ID")
        .loc[product_ids]
        .rename_axis("PRODUCT_ID")
        .reset_index()
        .copy()
    )

    # 4. Clean product text features
    for col in CONTENT_FEATURES:
        content_catalog[col] = (
            content_catalog[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({
                "NO COMMODITY DESCRIPTION": "",
                "NO SUBCOMMODITY DESCRIPTION": "",
            })
        )

    # 5. Combine product descriptions
    content_catalog["content_text"] = (
        content_catalog[CONTENT_FEATURES]
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 6. Validate content features
    if content_catalog["content_text"].eq("").any():
        raise ValueError("Empty product descriptions found.")

    if len(content_catalog) != len(product_ids):
        raise ValueError("Content catalog size mismatch.")

    return content_catalog



def prepare_content_categorical_features(
    content_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare brand and product size features."""

    # 1. Validate required columns
    required = {
        "PRODUCT_ID",
        "content_text",
        "BRAND",
        "CURR_SIZE_OF_PRODUCT",
    }

    missing = required - set(content_catalog.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if content_catalog.empty:
        raise ValueError("Content catalog is empty.")

    if not content_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate product IDs.")

    # 2. Create a copy
    prepared_catalog = content_catalog.copy()

    # 3. Prepare categorical features
    categorical_features = [
        "BRAND",
        "CURR_SIZE_OF_PRODUCT",
    ]

    for col in categorical_features:
        prepared_catalog[col] = (
            prepared_catalog[col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
            .replace("", "UNKNOWN")
        )

    # 4. Validate categorical features
    if prepared_catalog[
        categorical_features
    ].isna().any().any():
        raise ValueError("Missing categorical features.")

    if len(prepared_catalog) != len(content_catalog):
        raise ValueError("Product count changed.")

    return prepared_catalog
