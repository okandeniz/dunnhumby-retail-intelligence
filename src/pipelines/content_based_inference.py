
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from src.recommendations.content_based import (
    recommend_content_based,
)


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "content_based"
)


def run_content_based_inference(
    product_id: int,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend products using a saved Content-Based model."""

    # 1. Validate parameters
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    # 2. Load trained KNN model
    model = joblib.load(
        MODEL_DIR / "content_knn.joblib"
    )

    # 3. Load sparse content matrix
    content_matrix = load_npz(
        MODEL_DIR / "content_matrix.npz"
    ).tocsr()

    # 4. Load model metadata
    metadata = joblib.load(
        MODEL_DIR / "content_based_metadata.joblib"
    )

    # 5. Load product catalog
    content_catalog = pd.read_csv(
        MODEL_DIR / "content_product_catalog.csv",
        keep_default_na=False,
    )

    # 6. Restore product indexes
    product_ids = pd.Index(
        metadata["product_ids"]
    )

    # 7. Validate saved artifacts
    expected_shape = (
        metadata["n_products"],
        metadata["n_features"],
    )

    if content_matrix.shape != expected_shape:
        raise ValueError(
            "Content matrix dimensions do not match metadata."
        )

    if len(content_catalog) != len(product_ids):
        raise ValueError("Content catalog size mismatch.")

    if not product_ids.is_unique:
        raise ValueError("Duplicate product IDs.")

    if not content_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate catalog product IDs.")

    np.testing.assert_array_equal(
        content_catalog["PRODUCT_ID"].to_numpy(),
        product_ids.to_numpy(),
    )

    if model.n_features_in_ != content_matrix.shape[1]:
        raise ValueError(
            "Model feature dimensions do not match."
        )

    # 8. Generate recommendations
    recommendations = recommend_content_based(
        product_id=product_id,
        model=model,
        content_matrix=content_matrix,
        product_ids=product_ids,
        content_catalog=content_catalog,
        top_n=top_n,
    )

    # 9. Validate recommendations
    if product_id in recommendations["PRODUCT_ID"].values:
        raise ValueError("Selected product was recommended.")

    if not recommendations["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate recommendations found.")

    if len(recommendations) > top_n:
        raise ValueError("Too many recommendations.")

    scores = recommendations[
        "cosine_similarity"
    ].to_numpy()

    if not np.isfinite(scores).all():
        raise ValueError("Invalid similarity scores.")

    if ((scores < -1e-6) | (scores > 1 + 1e-6)).any():
        raise ValueError("Similarity scores are out of range.")

    if (np.diff(scores) > 1e-6).any():
        raise ValueError("Recommendations are not sorted.")

    return recommendations


def main() -> None:
    """Run Content-Based inference without retraining."""

    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Content-Based Product Recommendation"
    )

    parser.add_argument(
        "--product-id",
        type=int,
        default=826249,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    print("CONTENT-BASED INFERENCE PIPELINE")
    print("-" * 40)

    # 2. Generate product recommendations
    recommendations = run_content_based_inference(
        product_id=args.product_id,
        top_n=args.top_n,
    )

    print("\nINFERENCE COMPLETED")
    print("Selected Product ID:", args.product_id)
    print("Recommendations:", len(recommendations))

    print("\nPRODUCT RECOMMENDATIONS")
    print(
        recommendations.round(4).to_string(
            index=False
        )
    )

    print("\nSaved Model Inference: PASSED")

    print(
        "\nCONTENT-BASED INFERENCE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
