
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from collections import defaultdict
from collections.abc import Callable, Mapping
from functools import lru_cache


def fit_item_based_knn(
    customer_item_matrix: csr_matrix,
) -> tuple[NearestNeighbors, csr_matrix]:
    """Fit an Item-Based KNN model using cosine distance."""

    if customer_item_matrix.shape[0] < 2:
        raise ValueError("At least two customers are required.")

    if customer_item_matrix.shape[1] < 2:
        raise ValueError("At least two products are required.")

    # 1. Transpose customer-product matrix
    item_user_matrix = (
        customer_item_matrix.T.tocsr()
    )

    # 2. Initialize nearest neighbors
    item_knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    # 3. Fit item-based model
    item_knn.fit(item_user_matrix)

    return item_knn, item_user_matrix


def recommend_similar_items(
    product_id: int,
    model: NearestNeighbors,
    item_user_matrix: csr_matrix,
    product_ids: pd.Index,
    valid_catalog: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend similar products using Item-Based KNN."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    if item_user_matrix.shape[0] != len(product_ids):
        raise ValueError("Product index and matrix size mismatch.")

    if not valid_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate catalog product IDs.")

    # 1. Find selected product index
    product_idx = product_ids.get_indexer(
        [product_id]
    )[0]

    if product_idx == -1:
        raise ValueError("Product is not available in the model.")

    # 2. Find nearest products
    distances, indices = model.kneighbors(
        item_user_matrix[product_idx],
        n_neighbors=min(
            top_n + 1,
            len(product_ids),
        ),
    )

    # 3. Create recommendation results
    results = pd.DataFrame({
        "PRODUCT_ID": product_ids.take(indices[0]),
        "cosine_similarity": 1 - distances[0],
    })

    # 4. Exclude selected product
    results = results.loc[
        results["PRODUCT_ID"] != product_id
    ].head(top_n).copy()

    # 5. Add product descriptions
    results = results.merge(
        valid_catalog[
            ["PRODUCT_ID", "product_label"]
        ],
        on="PRODUCT_ID",
        how="left",
        validate="one_to_one",
    )

    if results["product_label"].isna().any():
        raise ValueError("Missing recommended product labels.")

    # 6. Return recommendations
    return results[
        [
            "PRODUCT_ID",
            "product_label",
            "cosine_similarity",
        ]
    ].reset_index(drop=True)


def build_customer_seed_products(
    train_items: pd.DataFrame,
    product_ids: pd.Index,
    n_seed_items: int = 10,
) -> dict[int, list[int]]:
    """Find the most recent unique products for each customer."""

    required = {
        "household_key",
        "PRODUCT_ID",
        "basket_day",
    }

    missing = required - set(train_items.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if n_seed_items < 1:
        raise ValueError("n_seed_items must be positive.")

    # 1. Keep products supported by the model
    historical_items = train_items.loc[
        train_items["PRODUCT_ID"].isin(product_ids),
        ["household_key", "PRODUCT_ID", "basket_day"],
    ]

    # 2. Find last purchase day for each product
    last_purchases = (
        historical_items
        .groupby(
            ["household_key", "PRODUCT_ID"],
            as_index=False,
        )
        .agg(last_day=("basket_day", "max"))
    )

    # 3. Sort from most recent to oldest
    last_purchases = last_purchases.sort_values(
        ["household_key", "last_day", "PRODUCT_ID"],
        ascending=[True, False, True],
    )

    # 4. Select recent unique products
    recent_products = (
        last_purchases
        .groupby("household_key")
        .head(n_seed_items)
    )

    # 5. Create customer seed dictionary
    customer_seeds = (
        recent_products
        .groupby("household_key")["PRODUCT_ID"]
        .agg(list)
        .to_dict()
    )

    return customer_seeds


def create_item_neighbor_lookup(
    model: NearestNeighbors,
    item_user_matrix: csr_matrix,
    product_ids: pd.Index,
    n_item_neighbors: int = 30,
) -> Callable:
    """Create a cached item-neighbor lookup function."""

    if n_item_neighbors < 1:
        raise ValueError("n_item_neighbors must be positive.")

    if item_user_matrix.shape[0] != len(product_ids):
        raise ValueError("Product index mismatch.")

    if model.n_features_in_ != item_user_matrix.shape[1]:
        raise ValueError("KNN feature dimension mismatch.")

    @lru_cache(maxsize=10000)
    def get_item_neighbors(
        product_id: int,
    ) -> tuple:
        """Return nearest products and similarity scores."""

        product_idx = product_ids.get_indexer(
            [product_id]
        )[0]

        if product_idx == -1:
            raise ValueError("Product is not available.")

        # Find nearest products
        distances, indices = model.kneighbors(
            item_user_matrix[product_idx],
            n_neighbors=min(
                n_item_neighbors + 1,
                len(product_ids),
            ),
        )

        neighbors = []

        for idx, distance in zip(
            indices[0],
            distances[0],
        ):
            neighbor_id = int(product_ids[idx])
            similarity = float(1 - distance)

            if neighbor_id != product_id and similarity > 0:
                neighbors.append(
                    (neighbor_id, similarity)
                )

        return tuple(neighbors)

    return get_item_neighbors


def recommend_item_based_customer(
    household_key: int,
    customer_item_matrix: csr_matrix,
    customer_ids: pd.Index,
    product_ids: pd.Index,
    customer_seed_products: Mapping[int, list[int]],
    get_item_neighbors: Callable,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend new products using Item-Based KNN."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    # 1. Find customer index
    customer_idx = customer_ids.get_indexer(
        [household_key]
    )[0]

    if customer_idx == -1:
        raise ValueError("Unknown customer.")

    # 2. Get recent purchased products
    seed_products = customer_seed_products.get(
        household_key,
        [],
    )

    # 3. Get all previously purchased products
    purchased_ids = set(
        product_ids.take(
            customer_item_matrix[customer_idx].indices
        )
    )

    # 4. Combine similarity scores
    scores = defaultdict(float)

    for product_id in seed_products:
        neighbors = get_item_neighbors(
            int(product_id)
        )

        for neighbor_id, similarity in neighbors:
            if neighbor_id not in purchased_ids:
                scores[neighbor_id] += similarity

    # 5. Rank by score and product ID
    top_products = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_n]

    # 6. Return product recommendations
    return pd.DataFrame(
        top_products,
        columns=[
            "PRODUCT_ID",
            "recommendation_score",
        ],
    )

