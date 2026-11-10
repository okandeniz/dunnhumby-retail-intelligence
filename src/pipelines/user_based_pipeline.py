
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

from src.recommendations.user_based import (
    fit_user_based_knn,
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

TRAIN_END_DAY = 621
MIN_CUSTOMERS = 25

EXAMPLE_CUSTOMER_ID = 1
TOP_N = 10
N_NEIGHBORS = 30


def main() -> None:
    """Train, save and validate the User-Based KNN model."""

    print("USER-BASED KNN TRAINING PIPELINE")
    print("-" * 40)

    # 1. Create model directory
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 2. Load training data
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

    # 4. Filter valid retail products
    train_items = train_items.loc[
        train_items["PRODUCT_ID"].isin(
            valid_catalog["PRODUCT_ID"]
        )
    ].copy()

    print("\nDATA PREPARATION COMPLETED")
    print("Training Interactions:", len(train_items))

    # 5. Build customer-product matrix
    (
        customer_item_matrix,
        customer_ids,
        product_ids,
    ) = build_customer_item_matrix(
        train_items=train_items,
        min_customers=MIN_CUSTOMERS,
    )

    # 6. Validate notebook reference counts
    assert customer_item_matrix.shape == (2481, 11603)
    assert customer_item_matrix.nnz == 917897

    print("\nCUSTOMER-PRODUCT MATRIX COMPLETED")
    print("Matrix Shape:", customer_item_matrix.shape)
    print("Non-Zero Interactions:", customer_item_matrix.nnz)
    print("Matrix Validation: PASSED")

    # 7. Fit User-Based KNN model
    user_knn = fit_user_based_knn(
        customer_item_matrix=customer_item_matrix,
    )

    print("\nUSER-BASED KNN TRAINING COMPLETED")
    print("Customers:", len(customer_ids))
    print("Products:", len(product_ids))

    # 8. Prepare model product catalog
    model_catalog = valid_catalog.loc[
        valid_catalog["PRODUCT_ID"].isin(product_ids),
        ["PRODUCT_ID", "product_label"],
    ].copy()

    if len(model_catalog) != len(product_ids):
        raise ValueError("Model product catalog mismatch.")

    # 9. Generate recommendations before saving
    original_predictions = recommend_user_based(
        household_key=EXAMPLE_CUSTOMER_ID,
        model=user_knn,
        customer_item_matrix=customer_item_matrix,
        customer_ids=customer_ids,
        product_ids=product_ids,
        valid_catalog=model_catalog,
        top_n=TOP_N,
        n_neighbors=N_NEIGHBORS,
    )

    # 10. Define artifact paths
    model_path = MODEL_DIR / "user_knn.joblib"

    matrix_path = (
        MODEL_DIR / "customer_item_matrix.npz"
    )

    metadata_path = (
        MODEL_DIR / "user_based_metadata.joblib"
    )

    catalog_path = MODEL_DIR / "product_catalog.csv"

    # 11. Save trained model
    joblib.dump(
        user_knn,
        model_path,
    )

    # 12. Save sparse customer-product matrix
    save_npz(
        matrix_path,
        customer_item_matrix,
    )

    # 13. Save model metadata
    metadata = {
        "model_name": "User-Based KNN",
        "customer_ids": customer_ids.tolist(),
        "product_ids": product_ids.tolist(),
        "min_customers": MIN_CUSTOMERS,
        "similarity_metric": "cosine",
        "training_end_day": TRAIN_END_DAY,
        "default_neighbors": N_NEIGHBORS,
        "n_customers": len(customer_ids),
        "n_products": len(product_ids),
    }

    joblib.dump(
        metadata,
        metadata_path,
    )

    # 14. Save product catalog
    model_catalog.to_csv(
        catalog_path,
        index=False,
    )

    print("\nMODEL ARTIFACTS SAVED")
    print("Output Directory:", MODEL_DIR)

    # 15. Reload saved artifacts
    loaded_model = joblib.load(model_path)

    loaded_matrix = load_npz(
        matrix_path
    )

    loaded_metadata = joblib.load(
        metadata_path
    )

    loaded_catalog = pd.read_csv(
        catalog_path
    )

    loaded_customer_ids = pd.Index(
        loaded_metadata["customer_ids"]
    )

    loaded_product_ids = pd.Index(
        loaded_metadata["product_ids"]
    )

    # 16. Validate loaded artifacts
    assert loaded_matrix.shape == (
        len(loaded_customer_ids),
        len(loaded_product_ids),
    )

    assert (loaded_matrix != customer_item_matrix).nnz == 0

    assert loaded_model.n_features_in_ == len(
        loaded_product_ids
    )

    np.testing.assert_array_equal(
        loaded_customer_ids.to_numpy(),
        customer_ids.to_numpy(),
    )

    np.testing.assert_array_equal(
        loaded_product_ids.to_numpy(),
        product_ids.to_numpy(),
    )

    # 17. Generate recommendations using saved model
    loaded_predictions = recommend_user_based(
        household_key=EXAMPLE_CUSTOMER_ID,
        model=loaded_model,
        customer_item_matrix=loaded_matrix,
        customer_ids=loaded_customer_ids,
        product_ids=loaded_product_ids,
        valid_catalog=loaded_catalog,
        top_n=TOP_N,
        n_neighbors=N_NEIGHBORS,
    )

    # 18. Compare predictions
    pd.testing.assert_frame_equal(
        original_predictions,
        loaded_predictions,
        check_dtype=False,
        rtol=1e-6,
        atol=1e-6,
    )

    # 19. Validate recommended products
    customer_idx = loaded_customer_ids.get_loc(
        EXAMPLE_CUSTOMER_ID
    )

    purchased_ids = set(
        loaded_product_ids.take(
            loaded_matrix[customer_idx].indices
        )
    )

    assert not set(
        loaded_predictions["PRODUCT_ID"]
    ).intersection(purchased_ids)

    assert loaded_predictions["PRODUCT_ID"].is_unique

    print("\nSAVED MODEL VALIDATION")
    print("Model Reload: PASSED")
    print("Matrix Reload: PASSED")
    print("Prediction Comparison: PASSED")
    print("Previously Purchased Product Check: PASSED")

    print("\nCUSTOMER RECOMMENDATIONS")
    print("Customer ID:", EXAMPLE_CUSTOMER_ID)
    print(
        "Previously Purchased Products:",
        loaded_matrix[customer_idx].nnz,
    )

    print(
        loaded_predictions.round(4).to_string(
            index=False
        )
    )

    print(
        "\nUSER-BASED KNN PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
