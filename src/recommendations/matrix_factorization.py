
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, NMF


def fit_truncated_svd(
    customer_item_matrix: csr_matrix,
    n_components: int = 50,
    n_iter: int = 7,
    random_state: int = 42,
) -> tuple[TruncatedSVD, np.ndarray, np.ndarray]:
    """Fit Truncated SVD using customer-product interactions."""

    # 1. Validate model parameters
    if n_components < 1 or n_iter < 1:
        raise ValueError("Model parameters must be positive.")

    if customer_item_matrix.ndim != 2:
        raise ValueError("Interaction matrix must be 2D.")

    n_customers, n_products = customer_item_matrix.shape

    if n_components >= min(n_customers, n_products):
        raise ValueError("Too many latent components.")

    if customer_item_matrix.nnz == 0:
        raise ValueError("Interaction matrix is empty.")

    # 2. Initialize Truncated SVD
    svd_model = TruncatedSVD(
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state,
    )

    # 3. Learn customer latent factors
    user_factors = svd_model.fit_transform(
        customer_item_matrix
    )

    # 4. Get product latent factors
    item_factors = svd_model.components_.T

    # 5. Validate factor dimensions
    if user_factors.shape != (
        n_customers,
        n_components,
    ):
        raise ValueError("Invalid user factor dimensions.")

    if item_factors.shape != (
        n_products,
        n_components,
    ):
        raise ValueError("Invalid item factor dimensions.")

    if not np.isfinite(user_factors).all():
        raise ValueError("Invalid user factor values.")

    if not np.isfinite(item_factors).all():
        raise ValueError("Invalid item factor values.")

    return svd_model, user_factors, item_factors


def recommend_svd(
    household_key: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    customer_item_matrix: csr_matrix,
    customer_ids: pd.Index,
    product_ids: pd.Index,
    valid_catalog: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend new products using Truncated SVD factors."""

    # 1. Validate parameters
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    if customer_item_matrix.shape != (
        len(customer_ids),
        len(product_ids),
    ):
        raise ValueError("Matrix and index dimensions do not match.")

    if user_factors.shape[0] != len(customer_ids):
        raise ValueError("User factor dimensions do not match.")

    if item_factors.shape[0] != len(product_ids):
        raise ValueError("Item factor dimensions do not match.")

    if user_factors.shape[1] != item_factors.shape[1]:
        raise ValueError("Latent dimensions do not match.")

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

    # 3. Calculate product interaction scores
    scores = np.asarray(
        user_factors[customer_idx] @ item_factors.T
    ).ravel().copy()

    # 4. Exclude previously purchased products
    purchased_indices = customer_item_matrix[
        customer_idx
    ].indices

    scores[purchased_indices] = -np.inf

    # 5. Keep finite and positive scores
    candidate_indices = np.flatnonzero(
        np.isfinite(scores) & (scores > 0)
    )

    # 6. Select highest-scoring products
    top_indices = candidate_indices[
        np.argsort(
            scores[candidate_indices]
        )[::-1][:top_n]
    ]

    # 7. Build recommendation results
    recommendations = pd.DataFrame({
        "PRODUCT_ID": product_ids.take(top_indices),
        "recommendation_score": scores[top_indices],
    })

    # 8. Add product descriptions
    recommendations = recommendations.merge(
        valid_catalog[
            ["PRODUCT_ID", "product_label"]
        ],
        on="PRODUCT_ID",
        how="left",
        validate="one_to_one",
    )

    if recommendations["product_label"].isna().any():
        raise ValueError("Missing product labels.")

    # 9. Return ranked recommendations
    return recommendations[
        [
            "PRODUCT_ID",
            "product_label",
            "recommendation_score",
        ]
    ].reset_index(drop=True)


def fit_nmf(
    customer_item_matrix: csr_matrix,
    n_components: int = 50,
    max_iter: int = 800,
    tol: float = 1e-4,
    random_state: int = 42,
) -> tuple[NMF, np.ndarray, np.ndarray]:
    """Fit NMF using non-negative customer-product interactions."""

    # 1. Validate model parameters
    if n_components < 1 or max_iter < 1:
        raise ValueError("Model parameters must be positive.")

    if tol <= 0:
        raise ValueError("tol must be positive.")

    n_customers, n_products = customer_item_matrix.shape

    if n_components > min(n_customers, n_products):
        raise ValueError("Too many latent components.")

    if customer_item_matrix.nnz == 0:
        raise ValueError("Interaction matrix is empty.")

    if not np.isfinite(customer_item_matrix.data).all():
        raise ValueError("Interaction matrix contains invalid values.")

    if (customer_item_matrix.data < 0).any():
        raise ValueError("NMF requires non-negative data.")

    # 2. Initialize NMF model
    nmf_model = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="cd",
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )

    # 3. Learn customer latent factors
    user_factors = nmf_model.fit_transform(
        customer_item_matrix
    )

    # 4. Extract product latent factors
    item_factors = nmf_model.components_.T

    # 5. Validate factor dimensions
    if user_factors.shape != (n_customers, n_components):
        raise ValueError("Invalid NMF user factor dimensions.")

    if item_factors.shape != (n_products, n_components):
        raise ValueError("Invalid NMF item factor dimensions.")

    # 6. Validate non-negative factors
    if not np.isfinite(user_factors).all():
        raise ValueError("Invalid NMF user factors.")

    if not np.isfinite(item_factors).all():
        raise ValueError("Invalid NMF item factors.")

    if (user_factors < 0).any():
        raise ValueError("Negative user factors found.")

    if (item_factors < 0).any():
        raise ValueError("Negative item factors found.")

    return nmf_model, user_factors, item_factors


def recommend_nmf(
    household_key: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    customer_item_matrix: csr_matrix,
    customer_ids: pd.Index,
    product_ids: pd.Index,
    valid_catalog: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend new products using NMF latent factors."""

    # NMF and SVD use the same factor-based ranking logic.
    return recommend_svd(
        household_key=household_key,
        user_factors=user_factors,
        item_factors=item_factors,
        customer_item_matrix=customer_item_matrix,
        customer_ids=customer_ids,
        product_ids=product_ids,
        valid_catalog=valid_catalog,
        top_n=top_n,
    )

