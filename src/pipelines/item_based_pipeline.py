
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz, save_npz

from src.data.database import get_engine
from src.data.recommendations import (
    load_basket_product_interactions,
    load_product_catalog,
    prepare_valid_catalog,
)

from src.features.recommendations import (
    build_customer_item_matrix,
)

from src.recommendations.item_based import (
    fit_item_based_knn,
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

TRAIN_END_DAY = 621
MIN_CUSTOMERS = 25

EXAMPLE_PRODUCT_ID = 826249
TOP_N = 5


def main() -> None:
    """Train, save and validate the Item-Based KNN model."""

    print("ITEM-BASED KNN TRAINING PIPELINE")
    print("-" * 40)

    # 1. Create model directory
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 2. Load training interactions and product catalog
    engine = get_engine()

    try:
        train_items = load_basket_product_interactions(
            engine=engine,
            end_day=TRAIN_END_DAY,
        )

        product_catalog = load_product_catalog(
            engine=engine,
        )
    finally:
        engine.dispose()

    # 3. Prepare valid product catalog
    valid_catalog = prepare_valid_catalog(
        product_catalog=product_catalog,
    )

    # 4. Keep valid retail products
    train_items = train_items.loc[
        train_items["PRODUCT_ID"].isin(
            valid_catalog["PRODUCT_ID"]
        )
    ].copy()

    print("\nDATA PREPARATION COMPLETED")
    print("Training Interactions:", len(train_items))
    print("Valid Products:", len(valid_catalog))

    # 5. Build customer-product matrix
    (
        customer_item_matrix,
        customer_ids,
        product_ids,
    ) = build_customer_item_matrix(
        train_items=train_items,
        min_customers=MIN_CUSTOMERS,
    )

    print("\nCUSTOMER-PRODUCT MATRIX COMPLETED")
    print("Matrix Shape:", customer_item_matrix.shape)
    print("Non-Zero Interactions:", customer_item_matrix.nnz)

    # 6. Verify notebook reference counts
    assert customer_item_matrix.shape == (2481, 11603)
    assert customer_item_matrix.nnz == 917897

    print("Matrix Validation: PASSED")

    # 7. Train Item-Based KNN
    item_knn, item_user_matrix = fit_item_based_knn(
        customer_item_matrix=customer_item_matrix,
    )

    print("\nITEM-BASED KNN TRAINING COMPLETED")
    print("Item-User Matrix:", item_user_matrix.shape)

    # 8. Prepare catalog for model products
    model_catalog = valid_catalog.loc[
        valid_catalog["PRODUCT_ID"].isin(product_ids),
        ["PRODUCT_ID", "product_label"],
    ].copy()

    if len(model_catalog) != len(product_ids):
        raise ValueError("Model catalog does not match product IDs.")

    # 9. Generate original model recommendations
    original_predictions = recommend_similar_items(
        product_id=EXAMPLE_PRODUCT_ID,
        model=item_knn,
        item_user_matrix=item_user_matrix,
        product_ids=product_ids,
        valid_catalog=model_catalog,
        top_n=TOP_N,
    )

    # 10. Define artifact paths
    model_path = MODEL_DIR / "item_knn.joblib"
    matrix_path = MODEL_DIR / "item_user_matrix.npz"
    metadata_path = MODEL_DIR / "item_based_metadata.joblib"
    catalog_path = MODEL_DIR / "product_catalog.csv"

    # 11. Save KNN model
    joblib.dump(item_knn, model_path)

    # 12. Save sparse item-user matrix
    save_npz(
        matrix_path,
        item_user_matrix,
    )

    # 13. Save model metadata
    metadata = {
        "model_name": "Item-Based KNN",
        "product_ids": product_ids.tolist(),
        "min_customers": MIN_CUSTOMERS,
        "similarity_metric": "cosine",
        "training_end_day": TRAIN_END_DAY,
        "n_customers": len(customer_ids),
        "n_products": len(product_ids),
    }

    joblib.dump(
        metadata,
        metadata_path,
    )

    # 14. Save product descriptions
    model_catalog.to_csv(
        catalog_path,
        index=False,
    )

    print("\nMODEL ARTIFACTS SAVED")

    # 15. Reload saved artifacts
    loaded_model = joblib.load(model_path)
    loaded_matrix = load_npz(matrix_path)
    loaded_metadata = joblib.load(metadata_path)

    loaded_catalog = pd.read_csv(
        catalog_path,
    )

    loaded_product_ids = pd.Index(
        loaded_metadata["product_ids"]
    )

    # 16. Validate loaded matrix and configuration
    assert loaded_matrix.shape == item_user_matrix.shape
    assert (loaded_matrix != item_user_matrix).nnz == 0

    np.testing.assert_array_equal(
        loaded_product_ids.to_numpy(),
        product_ids.to_numpy(),
    )

    # 17. Predict using the saved model
    loaded_predictions = recommend_similar_items(
        product_id=EXAMPLE_PRODUCT_ID,
        model=loaded_model,
        item_user_matrix=loaded_matrix,
        product_ids=loaded_product_ids,
        valid_catalog=loaded_catalog,
        top_n=TOP_N,
    )

    # 18. Compare original and loaded predictions
    pd.testing.assert_frame_equal(
        original_predictions,
        loaded_predictions,
        check_dtype=False,
        rtol=1e-6,
        atol=1e-6,
    )

    print("\nSAVED MODEL VALIDATION")
    print("Model Reload: PASSED")
    print("Matrix Reload: PASSED")
    print("Original Prediction Comparison: PASSED")

    print("\nPRODUCT RECOMMENDATIONS")
    print(
        loaded_predictions.round(4).to_string(
            index=False
        )
    )

    print(
        "\nITEM-BASED KNN PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
