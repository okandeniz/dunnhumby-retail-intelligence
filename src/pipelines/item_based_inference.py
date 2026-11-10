
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from src.recommendations.item_based import (
    recommend_similar_items,
)


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "item_based"
)


def run_item_based_inference(
    product_id: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend products using a saved Item-Based KNN model."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    # 1. Load trained model
    model = joblib.load(
        MODEL_DIR / "item_knn.joblib"
    )

    # 2. Load item-user matrix
    item_user_matrix = load_npz(
        MODEL_DIR / "item_user_matrix.npz"
    )

    # 3. Load model metadata
    metadata = joblib.load(
        MODEL_DIR / "item_based_metadata.joblib"
    )

    # 4. Load product catalog
    product_catalog = pd.read_csv(
        MODEL_DIR / "product_catalog.csv"
    )

    # 5. Restore product indexes
    product_ids = pd.Index(
        metadata["product_ids"]
    )

    # 6. Validate saved artifacts
    if item_user_matrix.shape[0] != len(product_ids):
        raise ValueError(
            "Product index and matrix size mismatch."
        )

    if item_user_matrix.shape[1] != metadata["n_customers"]:
        raise ValueError(
            "Customer count does not match metadata."
        )

    if len(product_ids) != metadata["n_products"]:
        raise ValueError(
            "Product count does not match metadata."
        )

    if not product_ids.is_unique:
        raise ValueError("Duplicate model product IDs.")

    if model.n_features_in_ != item_user_matrix.shape[1]:
        raise ValueError(
            "Saved model and feature dimensions do not match."
        )

    if set(product_catalog["PRODUCT_ID"]) != set(product_ids):
        raise ValueError(
            "Product catalog does not match model products."
        )

    # 7. Generate recommendations
    recommendations = recommend_similar_items(
        product_id=product_id,
        model=model,
        item_user_matrix=item_user_matrix,
        product_ids=product_ids,
        valid_catalog=product_catalog,
        top_n=top_n,
    )

    # 8. Validate recommendations
    if product_id in recommendations["PRODUCT_ID"].values:
        raise ValueError("Selected product was recommended.")

    if not recommendations["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate recommendations found.")

    if not recommendations[
        "cosine_similarity"
    ].between(0, 1).all():
        raise ValueError("Invalid similarity scores.")

    return recommendations


def main() -> None:
    """Run Item-Based KNN inference without retraining."""

    # 1. Read command-line arguments
    parser = argparse.ArgumentParser(
        description="Item-Based Product Recommendation"
    )

    parser.add_argument(
        "--product-id",
        type=int,
        default=826249,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    print("ITEM-BASED KNN INFERENCE PIPELINE")
    print("-" * 40)

    # 2. Generate predictions
    recommendations = run_item_based_inference(
        product_id=args.product_id,
        top_n=args.top_n,
    )

    print("\nINFERENCE COMPLETED")
    print("Selected Product ID:", args.product_id)
    print("Recommendations:", len(recommendations))

    print("\nPRODUCT RECOMMENDATIONS")
    print(recommendations.round(4).to_string(index=False))

    # 3. Validate the notebook's reference example
    if args.product_id == 826249 and args.top_n == 5:

        expected_ids = [
            1098066,
            1082185,
            981760,
            883404,
            1029743,
        ]

        expected_scores = [
            0.7644,
            0.6903,
            0.6770,
            0.6597,
            0.6312,
        ]

        np.testing.assert_array_equal(
            recommendations["PRODUCT_ID"].to_numpy(),
            expected_ids,
        )

        np.testing.assert_allclose(
            recommendations[
                "cosine_similarity"
            ].to_numpy(),
            expected_scores,
            rtol=0,
            atol=0.0001,
        )

        print("\nNotebook Reference Comparison: PASSED")

    print("\nSaved Model Inference: PASSED")

    print(
        "\nITEM-BASED KNN INFERENCE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
