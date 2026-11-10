
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix


def build_product_frequency(
    train_items: pd.DataFrame,
    valid_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate product frequency and support using training data."""

    # 1. Validate required columns
    required_items = {"BASKET_ID", "PRODUCT_ID"}

    required_catalog = {
        "PRODUCT_ID",
        "product_label",
        "DEPARTMENT",
    }

    if missing := required_items - set(train_items.columns):
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    if missing := required_catalog - set(valid_catalog.columns):
        raise ValueError(f"Missing catalog columns: {sorted(missing)}")

    if train_items.empty:
        raise ValueError("Training interactions are empty.")

    if not valid_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate catalog product IDs.")

    if train_items[list(required_items)].isna().any().any():
        raise ValueError("Missing training interaction values.")

    # 2. Count training baskets
    total_baskets = train_items["BASKET_ID"].nunique()

    # 3. Calculate unique basket count per product
    product_frequency = (
        train_items
        .groupby("PRODUCT_ID", as_index=False)
        .agg(
            basket_count=("BASKET_ID", "nunique")
        )
    )

    # 4. Calculate product support
    product_frequency["support"] = (
        product_frequency["basket_count"] / total_baskets
    )

    # 5. Add product information
    product_frequency = product_frequency.merge(
        valid_catalog[
            ["PRODUCT_ID", "product_label", "DEPARTMENT"]
        ],
        on="PRODUCT_ID",
        how="left",
        validate="one_to_one",
    )

    # 6. Validate product information
    if product_frequency[
        ["product_label", "DEPARTMENT"]
    ].isna().any().any():
        raise ValueError("Products missing from valid catalog.")

    if not product_frequency["support"].between(0, 1).all():
        raise ValueError("Invalid support values.")

    # 7. Sort by purchase frequency
    product_frequency = (
        product_frequency
        .sort_values("basket_count", ascending=False)
        .reset_index(drop=True)
    )

    return product_frequency


def evaluate_support_thresholds(
    train_items: pd.DataFrame,
    product_frequency: pd.DataFrame,
    thresholds: tuple[float, ...] = (
        0.01, 0.005, 0.002, 0.001
    ),
) -> pd.DataFrame:
    """Evaluate product coverage at different support thresholds."""

    required = {"PRODUCT_ID", "basket_count", "support"}

    if missing := required - set(product_frequency.columns):
        raise ValueError(f"Missing frequency columns: {sorted(missing)}")

    if not {"BASKET_ID", "PRODUCT_ID"}.issubset(train_items.columns):
        raise ValueError("Missing training interaction columns.")

    if train_items.empty or product_frequency.empty:
        raise ValueError("Input data must not be empty.")

    if not thresholds or any(
        not 0 < threshold <= 1 for threshold in thresholds
    ):
        raise ValueError("Support thresholds must be between 0 and 1.")

    # 1. Count all training baskets
    total_baskets = train_items["BASKET_ID"].nunique()

    threshold_results = []

    # 2. Evaluate each support threshold
    for min_support in thresholds:

        # Select eligible products
        frequent_products = product_frequency.loc[
            product_frequency["support"] >= min_support,
            "PRODUCT_ID",
        ]

        # Keep eligible basket-product interactions
        filtered_items = train_items.loc[
            train_items["PRODUCT_ID"].isin(frequent_products),
            ["BASKET_ID", "PRODUCT_ID"],
        ]

        # Count distinct eligible products in each basket
        basket_product_counts = (
            filtered_items
            .groupby("BASKET_ID")["PRODUCT_ID"]
            .nunique()
        )

        # Baskets containing at least two eligible products
        baskets_with_pairs = int(
            (basket_product_counts >= 2).sum()
        )

        # 3. Store threshold results
        threshold_results.append({
            "min_support": min_support,
            "min_basket_count": int(
                np.ceil(min_support * total_baskets)
            ),
            "eligible_products": len(frequent_products),
            "baskets_with_pairs": baskets_with_pairs,
            "basket_coverage_pct": (
                baskets_with_pairs / total_baskets * 100
            ),
        })

    # 4. Return threshold comparison
    return pd.DataFrame(threshold_results)


def build_customer_item_matrix(
    train_items: pd.DataFrame,
    min_customers: int = 25,
) -> tuple[csr_matrix, pd.Index, pd.Index]:
    """Build a binary customer-product interaction matrix."""

    # 1. Validate inputs
    required = {"household_key", "PRODUCT_ID"}
    missing = required - set(train_items.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if train_items.empty:
        raise ValueError("Training interactions are empty.")

    if min_customers < 1:
        raise ValueError("min_customers must be positive.")

    if train_items[list(required)].isna().any().any():
        raise ValueError("Missing customer or product IDs.")

    # 2. Create unique customer-product interactions
    customer_product = (
        train_items[
            ["household_key", "PRODUCT_ID"]
        ]
        .drop_duplicates()
    )

    # 3. Count unique customers per product
    product_customer_counts = (
        customer_product
        .groupby("PRODUCT_ID")["household_key"]
        .nunique()
    )

    # 4. Select eligible products
    eligible_products = product_customer_counts.loc[
        product_customer_counts >= min_customers
    ].index

    # 5. Filter interactions
    cf_interactions = customer_product.loc[
        customer_product["PRODUCT_ID"].isin(
            eligible_products
        )
    ].copy()

    if cf_interactions.empty:
        raise ValueError("No eligible interactions found.")

    # 6. Create customer and product indexes
    customer_ids = pd.Index(
        sorted(cf_interactions["household_key"].unique())
    )

    product_ids = pd.Index(
        sorted(cf_interactions["PRODUCT_ID"].unique())
    )

    # 7. Map IDs to matrix positions
    row_indices = customer_ids.get_indexer(
        cf_interactions["household_key"]
    )

    col_indices = product_ids.get_indexer(
        cf_interactions["PRODUCT_ID"]
    )

    # 8. Create sparse interaction matrix
    customer_item_matrix = csr_matrix(
        (
            np.ones(
                len(cf_interactions),
                dtype=np.float32,
            ),
            (row_indices, col_indices),
        ),
        shape=(
            len(customer_ids),
            len(product_ids),
        ),
        dtype=np.float32,
    )

    # 9. Validate matrix
    if customer_item_matrix.nnz != len(cf_interactions):
        raise ValueError("Unexpected matrix interaction count.")

    return customer_item_matrix, customer_ids, product_ids

