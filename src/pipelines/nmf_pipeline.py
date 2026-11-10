
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
    fit_nmf,
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

TRAIN_END_DAY = 621
MIN_CUSTOMERS = 25

N_COMPONENTS = 50
MAX_ITER = 800
TOL = 1e-4
RANDOM_STATE = 42

EXAMPLE_CUSTOMER_ID = 1
TOP_N = 10


def main() -> None:
    """Train, save and validate an NMF recommender."""

    print("NMF TRAINING PIPELINE")
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

    # 4. Keep valid retail interactions
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

    # 6. Validate notebook counts
    assert customer_item_matrix.shape == (2481, 11603)
    assert customer_item_matrix.nnz == 917897

    print("\nCUSTOMER-PRODUCT MATRIX COMPLETED")
    print("Matrix Shape:", customer_item_matrix.shape)
    print("Non-Zero Interactions:", customer_item_matrix.nnz)

    # 7. Train NMF model
    (
        nmf_model,
        user_factors,
        item_factors,
    ) = fit_nmf(
        customer_item_matrix=customer_item_matrix,
        n_components=N_COMPONENTS,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
    )

    # 8. Validate latent factors
    assert user_factors.shape == (2481, 50)
    assert item_factors.shape == (11603, 50)

    assert np.isfinite(user_factors).all()
    assert np.isfinite(item_factors).all()

    assert (user_factors >= 0).all()
    assert (item_factors >= 0).all()

    reconstruction_error = float(
        nmf_model.reconstruction_err_
    )

    # 9. Compare notebook reconstruction error
    np.testing.assert_allclose(
        reconstruction_error,
        855.6686,
        rtol=0,
        atol=0.005,
    )

    converged = (
        nmf_model.n_iter_ < nmf_model.max_iter
    )

    print("\nNMF TRAINING COMPLETED")
    print("User Factors:", user_factors.shape)
    print("Item Factors:", item_factors.shape)
    print("Iterations:", nmf_model.n_iter_)
    print(
        "Reconstruction Error:",
        round(reconstruction_error, 4),
    )
    print("Converged:", converged)

    if not converged:
        print(
            "WARNING: NMF reached the maximum "
            "iteration limit."
        )

    # 10. Prepare model product catalog
    model_catalog = valid_catalog.loc[
        valid_catalog["PRODUCT_ID"].isin(product_ids),
        ["PRODUCT_ID", "product_label"],
    ].copy()

    if len(model_catalog) != len(product_ids):
        raise ValueError("Model catalog size mismatch.")

    # 11. Generate recommendations before saving
    original_predictions = recommend_nmf(
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

    # 12. Compare notebook reference products
    expected_top5 = [
        1027569,
        1127831,
        981760,
        878996,
        866211,
    ]

    np.testing.assert_array_equal(
        original_predictions["PRODUCT_ID"]
        .head(5)
        .to_numpy(),
        expected_top5,
    )

    print("\nNOTEBOOK REFERENCE COMPARISON: PASSED")

    # 13. Define artifact paths
    model_path = MODEL_DIR / "nmf_model.joblib"
    factors_path = MODEL_DIR / "nmf_factors.npz"
    matrix_path = MODEL_DIR / "customer_item_matrix.npz"
    metadata_path = MODEL_DIR / "nmf_metadata.joblib"
    catalog_path = MODEL_DIR / "product_catalog.csv"

    # 14. Save trained NMF model
    joblib.dump(
        nmf_model,
        model_path,
    )

    # 15. Save latent factors and ID mappings
    np.savez_compressed(
        factors_path,
        user_factors=user_factors,
        item_factors=item_factors,
        customer_ids=customer_ids.to_numpy(),
        product_ids=product_ids.to_numpy(),
    )

    # 16. Save sparse customer-product matrix
    save_npz(
        matrix_path,
        customer_item_matrix,
    )

    # 17. Save model metadata
    metadata = {
        "model_name": "Non-negative Matrix Factorization",
        "n_components": N_COMPONENTS,
        "init": "nndsvda",
        "solver": "cd",
        "max_iter": MAX_ITER,
        "tol": TOL,
        "random_state": RANDOM_STATE,
        "min_customers": MIN_CUSTOMERS,
        "training_end_day": TRAIN_END_DAY,
        "n_iterations": int(nmf_model.n_iter_),
        "converged": bool(converged),
        "reconstruction_error": reconstruction_error,
        "n_customers": len(customer_ids),
        "n_products": len(product_ids),
    }

    joblib.dump(
        metadata,
        metadata_path,
    )

    # 18. Save product descriptions
    model_catalog.to_csv(
        catalog_path,
        index=False,
    )

    print("\nMODEL ARTIFACTS SAVED")
    print("Output Directory:", MODEL_DIR)

    # 19. Reload trained NMF model
    loaded_model = joblib.load(
        model_path
    )

    # 20. Reload latent factors and ID mappings
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

    # 21. Reload interaction matrix and metadata
    loaded_matrix = load_npz(
        matrix_path
    ).tocsr()

    loaded_metadata = joblib.load(
        metadata_path
    )

    loaded_catalog = pd.read_csv(
        catalog_path
    )

    # 22. Validate saved artifacts
    assert loaded_matrix.shape == (
        len(loaded_customer_ids),
        len(loaded_product_ids),
    )

    assert (loaded_matrix != customer_item_matrix).nnz == 0

    assert loaded_model.n_components == (
        loaded_metadata["n_components"]
    )

    assert loaded_model.n_iter_ == (
        loaded_metadata["n_iterations"]
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
        item_factors,
        rtol=0,
        atol=0,
    )

    np.testing.assert_allclose(
        loaded_item_factors,
        loaded_model.components_.T,
        rtol=0,
        atol=0,
    )

    # 23. Generate recommendations from saved artifacts
    loaded_predictions = recommend_nmf(
        household_key=EXAMPLE_CUSTOMER_ID,
        user_factors=loaded_user_factors,
        item_factors=loaded_item_factors,
        customer_item_matrix=loaded_matrix,
        customer_ids=loaded_customer_ids,
        product_ids=loaded_product_ids,
        valid_catalog=loaded_catalog,
        top_n=TOP_N,
    )

    # 24. Compare original and loaded predictions
    pd.testing.assert_frame_equal(
        original_predictions,
        loaded_predictions,
        check_dtype=False,
        rtol=1e-6,
        atol=1e-6,
    )

    # 25. Exclude previously purchased products
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
        "\nNMF TRAINING PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
