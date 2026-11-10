
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors

from collections.abc import Callable, Mapping
from functools import lru_cache

from src.recommendations.item_based import (
    recommend_item_based_customer,
)


def fit_content_tfidf(
    content_catalog: pd.DataFrame,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit TF-IDF features using product descriptions."""

    # 1. Validate text features
    if "content_text" not in content_catalog.columns:
        raise ValueError("Missing content_text column.")

    if content_catalog.empty:
        raise ValueError("Content catalog is empty.")

    if content_catalog["content_text"].isna().any():
        raise ValueError("Missing product descriptions.")

    # 2. Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    # 3. Fit and transform product descriptions
    content_matrix = tfidf.fit_transform(
        content_catalog["content_text"]
    ).tocsr()

    # 4. Validate matrix
    if content_matrix.shape[0] != len(content_catalog):
        raise ValueError("TF-IDF matrix row count mismatch.")

    if content_matrix.shape[1] == 0:
        raise ValueError("No TF-IDF features generated.")

    return tfidf, content_matrix


def fit_improved_content_model(
    content_catalog: pd.DataFrame,
) -> tuple[ColumnTransformer, NearestNeighbors, csr_matrix]:
    """Fit a weighted Content-Based recommendation model."""

    # 1. Validate required columns
    required = {
        "content_text",
        "BRAND",
        "CURR_SIZE_OF_PRODUCT",
    }

    missing = required - set(content_catalog.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if content_catalog.empty:
        raise ValueError("Content catalog is empty.")

    if content_catalog[list(required)].isna().any().any():
        raise ValueError("Missing content features.")

    # 2. Create preprocessing pipeline
    content_preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
                "content_text",
            ),
            (
                "brand",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                ["BRAND"],
            ),
            (
                "size",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                ["CURR_SIZE_OF_PRODUCT"],
            ),
        ],
        transformer_weights={
            "text": 2.0,
            "brand": 0.5,
            "size": 1.0,
        },
        sparse_threshold=1.0,
    )

    # 3. Fit and transform product features
    content_matrix = (
        content_preprocessor
        .fit_transform(content_catalog)
        .tocsr()
    )

    # 4. Validate feature matrix
    if content_matrix.shape[0] != len(content_catalog):
        raise ValueError("Content matrix size mismatch.")

    if content_matrix.nnz == 0:
        raise ValueError("Content matrix has no features.")

    # 5. Initialize Content-Based KNN
    content_knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    # 6. Fit similarity model
    content_knn.fit(content_matrix)

    return (
        content_preprocessor,
        content_knn,
        content_matrix,
    )


def recommend_content_based(
    product_id: int,
    model: NearestNeighbors,
    content_matrix: csr_matrix,
    product_ids: pd.Index,
    content_catalog: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend similar products using content features."""

    # 1. Validate parameters
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    if content_matrix.shape[0] != len(product_ids):
        raise ValueError("Product index and matrix size mismatch.")

    if not product_ids.is_unique:
        raise ValueError("Duplicate product IDs.")

    if not np.array_equal(
        content_catalog["PRODUCT_ID"].to_numpy(),
        product_ids.to_numpy(),
    ):
        raise ValueError("Catalog and product index order mismatch.")

    if model.n_features_in_ != content_matrix.shape[1]:
        raise ValueError("Model and feature dimensions mismatch.")

    # 2. Find product index
    product_idx = product_ids.get_indexer(
        [product_id]
    )[0]

    if product_idx == -1:
        raise ValueError("Product is not available in the model.")

    # 3. Find nearest products
    n_neighbors = min(
        top_n + 20,
        len(product_ids),
    )

    distances, indices = model.kneighbors(
        content_matrix[product_idx],
        n_neighbors=n_neighbors,
    )

    # 4. Calculate similarity scores
    recommendations = pd.DataFrame({
        "PRODUCT_ID": product_ids.take(indices[0]),
        "cosine_similarity": 1 - distances[0],
    })

    # 5. Exclude selected product and zero similarity
    recommendations = recommendations.loc[
        (recommendations["PRODUCT_ID"] != product_id)
        & (recommendations["cosine_similarity"] > 0)
    ]

    # 6. Rank by cosine similarity
    recommendations = (
        recommendations
        .sort_values(
            "cosine_similarity",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    # 7. Add readable product information
    recommendations = recommendations.merge(
        content_catalog[
            [
                "PRODUCT_ID",
                "product_label",
                "COMMODITY_DESC",
            ]
        ],
        on="PRODUCT_ID",
        how="left",
        validate="one_to_one",
    )

    if recommendations["product_label"].isna().any():
        raise ValueError("Missing product descriptions.")

    # 8. Return recommendations
    return recommendations[
        [
            "PRODUCT_ID",
            "product_label",
            "COMMODITY_DESC",
            "cosine_similarity",
        ]
    ].reset_index(drop=True)


def create_content_neighbor_lookup(
    model: NearestNeighbors,
    content_matrix: csr_matrix,
    product_ids: pd.Index,
    n_neighbors: int = 30,
) -> Callable:
    """Create a cached content similarity lookup."""

    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive.")

    if content_matrix.shape[0] != len(product_ids):
        raise ValueError("Content matrix and product IDs do not match.")

    if model.n_features_in_ != content_matrix.shape[1]:
        raise ValueError("Model feature dimensions do not match.")

    @lru_cache(maxsize=12000)
    def get_content_neighbors(
        product_id: int,
    ) -> tuple:
        """Find similar products using content features."""

        # 1. Find product position
        product_idx = product_ids.get_indexer(
            [product_id]
        )[0]

        if product_idx == -1:
            return tuple()

        # 2. Find nearest products
        distances, indices = model.kneighbors(
            content_matrix[product_idx],
            n_neighbors=min(
                n_neighbors + 1,
                len(product_ids),
            ),
        )

        # 3. Prepare neighbors and similarities
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

    return get_content_neighbors


def recommend_content_customer(
    household_key: int,
    customer_item_matrix: csr_matrix,
    customer_ids: pd.Index,
    product_ids: pd.Index,
    customer_seed_products: Mapping[int, list[int]],
    get_content_neighbors: Callable,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend products from purchase history and content similarity."""

    # Reuse the same customer-level ranking logic.
    # Only the product similarity source is different.
    return recommend_item_based_customer(
        household_key=household_key,
        customer_item_matrix=customer_item_matrix,
        customer_ids=customer_ids,
        product_ids=product_ids,
        customer_seed_products=customer_seed_products,
        get_item_neighbors=get_content_neighbors,
        top_n=top_n,
    )

