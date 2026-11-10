
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from src.recommendations.matrix_factorization import (
    recommend_svd,
)


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "svd"
)


def run_svd_inference(
    household_key: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend products using saved Truncated SVD factors."""

    # 1. Validate parameters
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    # 2. Load trained SVD model
    model = joblib.load(
        MODEL_DIR / "svd_model.joblib"
    )

    # 3. Load latent factors and ID mappings
    with np.load(
        MODEL_DIR / "svd_factors.npz",
        allow_pickle=False,
    ) as data:

        user_factors = data[
            "user_factors"
        ].copy()

        item_factors = data[
            "item_factors"
        ].copy()

        customer_ids = pd.Index(
            data["customer_ids"].copy()
        )

        product_ids = pd.Index(
            data["product_ids"].copy()
        )

    # 4. Load customer-product interaction matrix
    customer_item_matrix = load_npz(
        MODEL_DIR / "customer_item_matrix.npz"
    ).tocsr()

    # 5. Load model metadata
    metadata = joblib.load(
        MODEL_DIR / "svd_metadata.joblib"
    )

    # 6. Load product catalog
    product_catalog = pd.read_csv(
        MODEL_DIR / "product_catalog.csv"
    )

    # 7. Validate saved artifacts
    expected_shape = (
        metadata["n_customers"],
        metadata["n_products"],
    )

    if customer_item_matrix.shape != expected_shape:
        raise ValueError(
            "Interaction matrix dimensions do not match metadata."
        )

    if user_factors.shape != (
        len(customer_ids),
        metadata["n_components"],
    ):
        raise ValueError("User factor dimensions do not match.")

    if item_factors.shape != (
        len(product_ids),
        metadata["n_components"],
    ):
        raise ValueError("Item factor dimensions do not match.")

    if model.components_.shape != (
        metadata["n_components"],
        len(product_ids),
    ):
        raise ValueError("Saved SVD model dimensions do not match.")

    if not customer_ids.is_unique or not product_ids.is_unique:
        raise ValueError("Duplicate customer or product IDs.")

    if not product_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate catalog product IDs.")

    if set(product_catalog["PRODUCT_ID"]) != set(product_ids):
        raise ValueError("Product catalog does not match model.")

    # 8. Validate model and saved product factors
    np.testing.assert_allclose(
        item_factors,
        model.components_.T,
        rtol=1e-6,
        atol=1e-6,
    )

    # 9. Validate factor values
    if not np.isfinite(user_factors).all():
        raise ValueError("Invalid user factor values.")

    if not np.isfinite(item_factors).all():
        raise ValueError("Invalid item factor values.")

    # 10. Generate recommendations
    recommendations = recommend_svd(
        household_key=household_key,
        user_factors=user_factors,
        item_factors=item_factors,
        customer_item_matrix=customer_item_matrix,
        customer_ids=customer_ids,
        product_ids=product_ids,
        valid_catalog=product_catalog,
        top_n=top_n,
    )

    # 11. Validate previously purchased products
    customer_idx = customer_ids.get_indexer(
        [household_key]
    )[0]

    purchased_ids = set(
        product_ids.take(
            customer_item_matrix[customer_idx].indices
        )
    )

    recommended_ids = set(
        recommendations["PRODUCT_ID"]
    )

    if recommended_ids.intersection(purchased_ids):
        raise ValueError(
            "Previously purchased product was recommended."
        )

    if not recommendations["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate recommended products.")

    if len(recommendations) > top_n:
        raise ValueError("Too many recommendations.")

    scores = recommendations[
        "recommendation_score"
    ].to_numpy()

    if not np.isfinite(scores).all():
        raise ValueError("Invalid recommendation scores.")

    if (np.diff(scores) > 1e-6).any():
        raise ValueError("Recommendations are not sorted.")

    return recommendations


def main() -> None:
    """Run Truncated SVD inference without retraining."""

    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Truncated SVD Product Recommendation"
    )

    parser.add_argument(
        "--household-key",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    print("TRUNCATED SVD INFERENCE PIPELINE")
    print("-" * 40)

    # 2. Generate recommendations
    recommendations = run_svd_inference(
        household_key=args.household_key,
        top_n=args.top_n,
    )

    print("\nINFERENCE COMPLETED")
    print("Customer ID:", args.household_key)
    print("Recommendations:", len(recommendations))

    print("\nCUSTOMER RECOMMENDATIONS")
    print(
        recommendations.round(4).to_string(
            index=False
        )
    )

    # 3. Validate notebook reference example
    if args.household_key == 1 and args.top_n >= 5:

        expected_top5 = [
            981760,
            962568,
            1127831,
            878996,
            1024306,
        ]

        np.testing.assert_array_equal(
            recommendations["PRODUCT_ID"]
            .head(5)
            .to_numpy(),
            expected_top5,
        )

        print("\nNotebook Reference Comparison: PASSED")

    print("Saved Model Inference: PASSED")

    print(
        "\nTRUNCATED SVD INFERENCE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
