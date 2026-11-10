
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def fit_user_based_knn(
    customer_item_matrix: csr_matrix,
) -> NearestNeighbors:
    """Fit a User-Based KNN model using cosine distance."""

    if customer_item_matrix.shape[0] < 2:
        raise ValueError("At least two customers are required.")

    if customer_item_matrix.shape[1] < 1:
        raise ValueError("At least one product is required.")

    # 1. Initialize User-Based KNN
    user_knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    # 2. Fit customer-product matrix
    user_knn.fit(customer_item_matrix)

    return user_knn


def recommend_user_based(
    household_key: int,
    model: NearestNeighbors,
    customer_item_matrix: csr_matrix,
    customer_ids: pd.Index,
    product_ids: pd.Index,
    valid_catalog: pd.DataFrame,
    top_n: int = 5,
    n_neighbors: int = 30,
) -> pd.DataFrame:
    """Recommend products using similar customers."""

    # 1. Validate parameters
    if top_n < 1 or n_neighbors < 1:
        raise ValueError(
            "top_n and n_neighbors must be positive."
        )

    if customer_item_matrix.shape != (
        len(customer_ids),
        len(product_ids),
    ):
        raise ValueError("Matrix and index dimensions do not match.")

    if not customer_ids.is_unique or not product_ids.is_unique:
        raise ValueError("Duplicate customer or product IDs.")

    if not valid_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate catalog product IDs.")

    # 2. Find customer index
    customer_idx = customer_ids.get_indexer(
        [household_key]
    )[0]

    if customer_idx == -1:
        raise ValueError(
            "Customer is not available in the model."
        )

    # 3. Find similar customers
    distances, indices = model.kneighbors(
        customer_item_matrix[customer_idx],
        n_neighbors=min(
            n_neighbors + 1,
            len(customer_ids),
        ),
    )

    neighbor_indices = indices[0]
    similarities = 1 - distances[0]

    # 4. Exclude target customer and zero-similarity neighbors
    mask = (
        (neighbor_indices != customer_idx)
        & (similarities > 0)
    )

    neighbor_indices = neighbor_indices[mask]
    similarities = similarities[mask]

    columns = [
        "PRODUCT_ID",
        "product_label",
        "recommendation_score",
    ]

    if len(neighbor_indices) == 0:
        return pd.DataFrame(columns=columns)

    # 5. Calculate similarity-weighted product scores
    neighbor_purchases = customer_item_matrix[
        neighbor_indices
    ]

    scores = np.asarray(
        neighbor_purchases.T @ similarities
    ).ravel()

    scores = scores / similarities.sum()

    # 6. Exclude previously purchased products
    purchased_indices = customer_item_matrix[
        customer_idx
    ].indices

    scores[purchased_indices] = 0

    # 7. Select products with positive scores
    candidate_indices = np.flatnonzero(
        scores > 0
    )

    top_indices = candidate_indices[
        np.argsort(
            scores[candidate_indices]
        )[::-1][:top_n]
    ]

    # 8. Build recommendation results
    recommendations = pd.DataFrame({
        "PRODUCT_ID": product_ids.take(top_indices),
        "recommendation_score": scores[top_indices],
    })

    # 9. Add readable product labels
    recommendations = recommendations.merge(
        valid_catalog[
            ["PRODUCT_ID", "product_label"]
        ],
        on="PRODUCT_ID",
        how="left",
        validate="one_to_one",
    )

    if recommendations["product_label"].isna().any():
        raise ValueError("Missing recommended product labels.")

    # 10. Return recommendations
    return recommendations[
        columns
    ].reset_index(drop=True)
