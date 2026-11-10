
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from src.recommendations.user_based import (
    recommend_user_based,
)


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "user_based"
)


def run_user_based_inference(
    household_key: int,
    top_n: int = 10,
    n_neighbors: int = 30,
) -> pd.DataFrame:
    """Recommend products using a saved User-Based KNN model."""

    # 1. Validate parameters
    if top_n < 1 or n_neighbors < 1:
        raise ValueError(
            "top_n and n_neighbors must be positive."
        )

    # 2. Load trained model
    model = joblib.load(
        MODEL_DIR / "user_knn.joblib"
    )

    # 3. Load customer-product matrix
    customer_item_matrix = load_npz(
        MODEL_DIR / "customer_item_matrix.npz"
    ).tocsr()

    # 4. Load model metadata
    metadata = joblib.load(
        MODEL_DIR / "user_based_metadata.joblib"
    )

    # 5. Load product catalog
    product_catalog = pd.read_csv(
        MODEL_DIR / "product_catalog.csv"
    )

    # 6. Restore customer and product IDs
    customer_ids = pd.Index(
        metadata["customer_ids"]
    )

    product_ids = pd.Index(
        metadata["product_ids"]
    )

    # 7. Validate saved artifacts
    expected_shape = (
        metadata["n_customers"],
        metadata["n_products"],
    )

    if customer_item_matrix.shape != expected_shape:
        raise ValueError(
            "Saved matrix dimensions do not match metadata."
        )

    if len(customer_ids) != expected_shape[0]:
        raise ValueError("Customer index mismatch.")

    if len(product_ids) != expected_shape[1]:
        raise ValueError("Product index mismatch.")

    if model.n_features_in_ != len(product_ids):
        raise ValueError(
            "Model feature dimensions do not match."
        )

    if not customer_ids.is_unique or not product_ids.is_unique:
        raise ValueError("Duplicate customer or product IDs.")

    if not product_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate catalog product IDs.")

    if set(product_catalog["PRODUCT_ID"]) != set(product_ids):
        raise ValueError("Product catalog mismatch.")

    # 8. Generate customer recommendations
    recommendations = recommend_user_based(
        household_key=household_key,
        model=model,
        customer_item_matrix=customer_item_matrix,
        customer_ids=customer_ids,
        product_ids=product_ids,
        valid_catalog=product_catalog,
        top_n=top_n,
        n_neighbors=n_neighbors,
    )

    # 9. Validate recommendations
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

    if ((scores < -1e-6) | (scores > 1 + 1e-6)).any():
        raise ValueError("Scores are outside the valid range.")

    if (np.diff(scores) > 1e-6).any():
        raise ValueError("Recommendations are not sorted.")

    return recommendations


def main() -> None:
    """Run User-Based KNN inference without retraining."""

    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="User-Based Product Recommendation"
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

    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    print("USER-BASED KNN INFERENCE PIPELINE")
    print("-" * 40)

    # 2. Generate recommendations
    recommendations = run_user_based_inference(
        household_key=args.household_key,
        top_n=args.top_n,
        n_neighbors=args.n_neighbors,
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

    # 3. Compare the notebook reference example
    if (
        args.household_key == 1
        and args.top_n == 10
        and args.n_neighbors == 30
    ):
        expected_ids = [
            1127831,
            854852,
            1024306,
            962568,
            981760,
            866211,
            878996,
            834484,
            1023720,
            1007195,
        ]

        np.testing.assert_array_equal(
            recommendations["PRODUCT_ID"].to_numpy(),
            expected_ids,
        )

        print("\nNotebook Reference Comparison: PASSED")

    print("Saved Model Inference: PASSED")

    print(
        "\nUSER-BASED KNN INFERENCE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
