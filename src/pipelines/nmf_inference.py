
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from src.recommendations.matrix_factorization import (
    recommend_nmf,
)


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "nmf"
)


def run_nmf_inference(
    household_key: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend products using saved NMF latent factors."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    # 1. Load trained NMF model
    model = joblib.load(
        MODEL_DIR / "nmf_model.joblib"
    )

    # 2. Load latent factors and ID mappings
    with np.load(
        MODEL_DIR / "nmf_factors.npz",
        allow_pickle=False,
    ) as data:
        user_factors = data["user_factors"].copy()
        item_factors = data["item_factors"].copy()

        customer_ids = pd.Index(
            data["customer_ids"].copy()
        )

        product_ids = pd.Index(
            data["product_ids"].copy()
        )

    # 3. Load customer-product interaction matrix
    customer_item_matrix = load_npz(
        MODEL_DIR / "customer_item_matrix.npz"
    ).tocsr()

    # 4. Load model metadata
    metadata = joblib.load(
        MODEL_DIR / "nmf_metadata.joblib"
    )

    # 5. Load product catalog
    product_catalog = pd.read_csv(
        MODEL_DIR / "product_catalog.csv"
    )

    # 6. Validate artifact dimensions
    expected_shape = (
        metadata["n_customers"],
        metadata["n_products"],
    )

    if customer_item_matrix.shape != expected_shape:
        raise ValueError("Interaction matrix size mismatch.")

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
        raise ValueError("Saved NMF model dimensions do not match.")

    if len(customer_ids) != metadata["n_customers"]:
        raise ValueError("Customer index size mismatch.")

    if len(product_ids) != metadata["n_products"]:
        raise ValueError("Product index size mismatch.")

    # 7. Validate IDs and product catalog
    if not customer_ids.is_unique or not product_ids.is_unique:
        raise ValueError("Duplicate customer or product IDs.")

    if not product_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate catalog product IDs.")

    if set(product_catalog["PRODUCT_ID"]) != set(product_ids):
        raise ValueError("Product catalog does not match model.")

    if household_key not in customer_ids:
        raise ValueError("Customer is not available in the model.")

    # 8. Validate saved latent factors
    np.testing.assert_allclose(
        item_factors,
        model.components_.T,
        rtol=1e-6,
        atol=1e-6,
    )

    if not np.isfinite(user_factors).all():
        raise ValueError("Invalid user factor values.")

    if not np.isfinite(item_factors).all():
        raise ValueError("Invalid item factor values.")

    if (user_factors < 0).any() or (item_factors < 0).any():
        raise ValueError("NMF factors must be non-negative.")

    # 9. Generate recommendations without retraining
    recommendations = recommend_nmf(
        household_key=household_key,
        user_factors=user_factors,
        item_factors=item_factors,
        customer_item_matrix=customer_item_matrix,
        customer_ids=customer_ids,
        product_ids=product_ids,
        valid_catalog=product_catalog,
        top_n=top_n,
    )

    # 10. Check previously purchased products
    customer_idx = customer_ids.get_loc(
        household_key
    )

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

    # 11. Validate recommendation results
    if not recommendations["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate recommendations found.")

    if len(recommendations) > top_n:
        raise ValueError("Too many recommendations.")

    scores = recommendations[
        "recommendation_score"
    ].to_numpy()

    if not np.isfinite(scores).all():
        raise ValueError("Invalid recommendation scores.")

    if (scores < 0).any():
        raise ValueError("Negative NMF recommendation scores.")

    if (np.diff(scores) > 1e-6).any():
        raise ValueError("Recommendations are not sorted.")

    return recommendations


def main() -> None:
    """Run NMF inference without retraining."""

    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="NMF Product Recommendation"
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

    print("NMF INFERENCE PIPELINE")
    print("-" * 40)

    # 2. Generate recommendations
    recommendations = run_nmf_inference(
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

    # 3. Compare notebook reference example
    if args.household_key == 1 and args.top_n >= 5:

        expected_top5 = [
            1027569,
            1127831,
            981760,
            878996,
            866211,
        ]

        np.testing.assert_array_equal(
            recommendations["PRODUCT_ID"]
            .head(5)
            .to_numpy(),
            expected_top5,
        )

        print("\nNotebook Reference Comparison: PASSED")

    # 4. Report stored convergence status
    metadata = joblib.load(
        MODEL_DIR / "nmf_metadata.joblib"
    )

    print(
        "\nNMF Converged:",
        metadata["converged"],
    )

    if not metadata["converged"]:
        print(
            "WARNING: NMF training did not confirm convergence."
        )

    print("\nSaved Model Inference: PASSED")

    print(
        "\nNMF INFERENCE PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
