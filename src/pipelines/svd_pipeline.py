
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

from src.recommendations.matrix_factorization import (
    fit_truncated_svd,
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

TRAIN_END_DAY = 621
MIN_CUSTOMERS = 25

N_COMPONENTS = 50
N_ITER = 7
RANDOM_STATE = 42

EXAMPLE_CUSTOMER_ID = 1
TOP_N = 10


def main() -> None:
    """Train, save and validate a Truncated SVD recommender."""

    print("TRUNCATED SVD TRAINING PIPELINE")
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

    # 3. Prepare valid retail catalog
    valid_catalog = prepare_valid_catalog(
        product_catalog=product_catalog,
    )

    # 4. Filter training interactions
    train_items = train_items.loc[
        train_items["PRODUCT_ID"].isin(
            valid_catalog["PRODUCT_ID"]
        )
    ].copy()

    print("\nDATA PREPARATION COMPLETED")
    print("Training Interactions:", len(train_items))

    # 5. Build customer-product interaction matrix
    (
        customer_item_matrix,
        customer_ids,
        product_ids,
    ) = build_customer_item_matrix(
        train_items=train_items,
        min_customers=MIN_CUSTOMERS,
    )

    # 6. Validate reference counts
    assert customer_item_matrix.shape == (2481, 11603)
    assert customer_item_matrix.nnz == 917897

    print("\nCUSTOMER-PRODUCT MATRIX COMPLETED")
    print("Matrix Shape:", customer_item_matrix.shape)
    print("Non-Zero Interactions:", customer_item_matrix.nnz)
    print("Matrix Validation: PASSED")

    # 7. Train Truncated SVD
    (
        svd_model,
        user_factors,
        item_factors,
    ) = fit_truncated_svd(
        customer_item_matrix=customer_item_matrix,
        n_components=N_COMPONENTS,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
    )

    # 8. Validate latent factors
    assert user_factors.shape == (2481, 50)
    assert item_factors.shape == (11603, 50)

    explained_variance = float(
        svd_model.explained_variance_ratio_.sum()
    )

    np.testing.assert_allclose(
        explained_variance,
        0.1563,
        rtol=0,
        atol=0.0001,
    )

    print("\nTRUNCATED SVD TRAINING COMPLETED")
    print("User Factors:", user_factors.shape)
    print("Item Factors:", item_factors.shape)
    print(
        "Explained Variance Ratio:",
        round(explained_variance, 4),
    )

    # 9. Prepare product catalog
    model_catalog = valid_catalog.loc[
        valid_catalog["PRODUCT_ID"].isin(product_ids),
        ["PRODUCT_ID", "product_label"],
    ].copy()

    if len(model_catalog) != len(product_ids):
        raise ValueError("Model catalog size mismatch.")

    # 10. Generate recommendations before saving
    original_predictions = recommend_svd(
        household_key=EXAMPLE_CUSTOMER_ID,
        user_factors=user_factors,
        item_factors=item_factors,
        customer_item_matrix=customer_item_matrix,
        customer_ids=customer_ids,
        product_ids=product_ids,
        valid_catalog=model_catalog,
        top_n=TOP_N,
    )

    assert len(original_predictions) == TOP_N

    # 11. Validate notebook reference product IDs
    expected_top5 = [
        981760,
        962568,
        1127831,
        878996,
        1024306,
    ]

    np.testing.assert_array_equal(
        original_predictions["PRODUCT_ID"]
        .head(5)
        .to_numpy(),
        expected_top5,
    )

    print("\nNOTEBOOK REFERENCE COMPARISON: PASSED")

    # 12. Define artifact paths
    model_path = MODEL_DIR / "svd_model.joblib"
    factors_path = MODEL_DIR / "svd_factors.npz"

    matrix_path = (
        MODEL_DIR / "customer_item_matrix.npz"
    )

    metadata_path = MODEL_DIR / "svd_metadata.joblib"
    catalog_path = MODEL_DIR / "product_catalog.csv"

    # 13. Save trained SVD model
    joblib.dump(
        svd_model,
        model_path,
    )

    # 14. Save latent factors and ID mappings
    np.savez_compressed(
        factors_path,
        user_factors=user_factors,
        item_factors=item_factors,
        customer_ids=customer_ids.to_numpy(),
        product_ids=product_ids.to_numpy(),
    )

    # 15. Save sparse interaction matrix
    save_npz(
        matrix_path,
        customer_item_matrix,
    )

    # 16. Save model metadata
    metadata = {
        "model_name": "Truncated SVD",
        "n_components": N_COMPONENTS,
        "n_iter": N_ITER,
        "random_state": RANDOM_STATE,
        "min_customers": MIN_CUSTOMERS,
        "training_end_day": TRAIN_END_DAY,
        "explained_variance_ratio": explained_variance,
        "n_customers": len(customer_ids),
        "n_products": len(product_ids),
    }

    joblib.dump(
        metadata,
        metadata_path,
    )

    # 17. Save product catalog
    model_catalog.to_csv(
        catalog_path,
        index=False,
    )

    print("\nMODEL ARTIFACTS SAVED")
    print("Output Directory:", MODEL_DIR)

    # 18. Reload trained model
    loaded_model = joblib.load(
        model_path
    )

    # 19. Reload latent factors
    with np.load(
        factors_path,
        allow_pickle=False,
    ) as data:
        loaded_user_factors = data[
            "user_factors"
        ].copy()

        loaded_item_factors = data[
            "item_factors"
        ].copy()

        loaded_customer_ids = pd.Index(
            data["customer_ids"].copy()
        )

        loaded_product_ids = pd.Index(
            data["product_ids"].copy()
        )

    # 20. Reload interaction matrix and metadata
    loaded_matrix = load_npz(
        matrix_path
    ).tocsr()

    loaded_metadata = joblib.load(
        metadata_path
    )

    loaded_catalog = pd.read_csv(
        catalog_path
    )

    # 21. Validate loaded artifacts
    assert loaded_matrix.shape == (
        len(loaded_customer_ids),
        len(loaded_product_ids),
    )

    assert (loaded_matrix != customer_item_matrix).nnz == 0

    assert loaded_model.n_components == (
        loaded_metadata["n_components"]
    )

    np.testing.assert_array_equal(
        loaded_customer_ids.to_numpy(),
        customer_ids.to_numpy(),
    )

    np.testing.assert_array_equal(
        loaded_product_ids.to_numpy(),
        product_ids.to_numpy(),
    )

    np.testing.assert_allclose(
        loaded_user_factors,
        user_factors,
        rtol=0,
        atol=0,
    )

    np.testing.assert_allclose(
        loaded_item_factors,
        loaded_model.components_.T,
        rtol=0,
        atol=0,
    )

    # 22. Generate recommendations using loaded artifacts
    loaded_predictions = recommend_svd(
        household_key=EXAMPLE_CUSTOMER_ID,
        user_factors=loaded_user_factors,
        item_factors=loaded_item_factors,
        customer_item_matrix=loaded_matrix,
        customer_ids=loaded_customer_ids,
        product_ids=loaded_product_ids,
        valid_catalog=loaded_catalog,
        top_n=TOP_N,
    )

    # 23. Compare predictions before and after saving
    pd.testing.assert_frame_equal(
        original_predictions,
        loaded_predictions,
        check_dtype=False,
        rtol=1e-6,
        atol=1e-6,
    )

    # 24. Verify that purchased products are excluded
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
    print("Latent Factors Reload: PASSED")
    print("Interaction Matrix Reload: PASSED")
    print("Prediction Comparison: PASSED")
    print("Previously Purchased Product Check: PASSED")

    print("\nCUSTOMER RECOMMENDATIONS")
    print("Customer ID:", EXAMPLE_CUSTOMER_ID)

    print(
        loaded_predictions.round(4).to_string(
            index=False
        )
    )

    print(
        "\nTRUNCATED SVD PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
