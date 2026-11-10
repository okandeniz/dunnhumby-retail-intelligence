
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

from src.features.content_based import (
    prepare_content_catalog,
    prepare_content_categorical_features,
)

from src.recommendations.content_based import (
    fit_improved_content_model,
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

TRAIN_END_DAY = 621
MIN_CUSTOMERS = 25

EXAMPLE_PRODUCT_ID = 826249
TOP_N = 10


def main() -> None:
    """Train, save and validate the Content-Based model."""

    print("CONTENT-BASED TRAINING PIPELINE")
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

    # 4. Filter training interactions
    train_items = train_items.loc[
        train_items["PRODUCT_ID"].isin(
            valid_catalog["PRODUCT_ID"]
        )
    ].copy()

    print("\nDATA PREPARATION COMPLETED")
    print("Training Interactions:", len(train_items))

    # 5. Obtain eligible products from training data
    (
        customer_item_matrix,
        customer_ids,
        product_ids,
    ) = build_customer_item_matrix(
        train_items=train_items,
        min_customers=MIN_CUSTOMERS,
    )

    assert customer_item_matrix.shape == (2481, 11603)
    assert customer_item_matrix.nnz == 917897

    print("\nPRODUCT SELECTION COMPLETED")
    print("Eligible Products:", len(product_ids))

    # 6. Prepare content catalog
    content_catalog = prepare_content_catalog(
        valid_catalog=valid_catalog,
        product_ids=product_ids,
    )

    # 7. Prepare categorical features
    content_catalog = prepare_content_categorical_features(
        content_catalog=content_catalog,
    )

    content_product_ids = pd.Index(
        content_catalog["PRODUCT_ID"]
    )

    np.testing.assert_array_equal(
        content_product_ids.to_numpy(),
        product_ids.to_numpy(),
    )

    print("\nCONTENT FEATURE PREPARATION COMPLETED")
    print("Content Products:", len(content_catalog))

    # 8. Fit improved Content-Based model
    (
        content_preprocessor,
        content_knn,
        content_matrix,
    ) = fit_improved_content_model(
        content_catalog=content_catalog,
    )

    # 9. Validate notebook reference values
    assert content_matrix.shape == (11603, 4712)
    assert content_matrix.nnz == 156268

    print("\nCONTENT-BASED MODEL TRAINING COMPLETED")
    print("Content Matrix:", content_matrix.shape)
    print("Non-Zero Values:", content_matrix.nnz)
    print("Feature Matrix Validation: PASSED")

    # 10. Generate recommendations before saving
    original_predictions = recommend_content_based(
        product_id=EXAMPLE_PRODUCT_ID,
        model=content_knn,
        content_matrix=content_matrix,
        product_ids=content_product_ids,
        content_catalog=content_catalog,
        top_n=TOP_N,
    )

    assert len(original_predictions) == TOP_N

    print("\nORIGINAL MODEL PREDICTIONS COMPLETED")

    # 11. Define artifact paths
    preprocessor_path = (
        MODEL_DIR / "content_preprocessor.joblib"
    )

    model_path = (
        MODEL_DIR / "content_knn.joblib"
    )

    matrix_path = (
        MODEL_DIR / "content_matrix.npz"
    )

    metadata_path = (
        MODEL_DIR / "content_based_metadata.joblib"
    )

    catalog_path = (
        MODEL_DIR / "content_product_catalog.csv"
    )

    # 12. Save fitted preprocessor and KNN model
    joblib.dump(
        content_preprocessor,
        preprocessor_path,
    )

    joblib.dump(
        content_knn,
        model_path,
    )

    # 13. Save sparse content matrix
    save_npz(
        matrix_path,
        content_matrix,
    )

    # 14. Save model metadata
    metadata = {
        "model_name": "Improved Content-Based KNN",
        "product_ids": content_product_ids.tolist(),
        "training_end_day": TRAIN_END_DAY,
        "min_customers": MIN_CUSTOMERS,
        "similarity_metric": "cosine",
        "feature_groups": [
            "content_text",
            "BRAND",
            "CURR_SIZE_OF_PRODUCT",
        ],
        "feature_weights": {
            "text": 2.0,
            "brand": 0.5,
            "size": 1.0,
        },
        "n_products": len(content_product_ids),
        "n_features": content_matrix.shape[1],
    }

    joblib.dump(
        metadata,
        metadata_path,
    )

    # 15. Save product information in matrix row order
    content_catalog[
        [
            "PRODUCT_ID",
            "product_label",
            "COMMODITY_DESC",
            "content_text",
            "BRAND",
            "CURR_SIZE_OF_PRODUCT",
        ]
    ].to_csv(
        catalog_path,
        index=False,
    )

    print("\nMODEL ARTIFACTS SAVED")
    print("Output Directory:", MODEL_DIR)

    # 16. Reload saved artifacts
    loaded_preprocessor = joblib.load(
        preprocessor_path
    )

    loaded_model = joblib.load(
        model_path
    )

    loaded_matrix = load_npz(
        matrix_path
    ).tocsr()

    loaded_metadata = joblib.load(
        metadata_path
    )

    loaded_catalog = pd.read_csv(
        catalog_path,
        keep_default_na=False,
    )

    loaded_product_ids = pd.Index(
        loaded_metadata["product_ids"]
    )

    # 17. Validate loaded artifacts
    assert loaded_matrix.shape == content_matrix.shape

    assert (loaded_matrix != content_matrix).nnz == 0

    assert loaded_model.n_features_in_ == (
        loaded_matrix.shape[1]
    )

    assert loaded_metadata["n_products"] == (
        len(loaded_product_ids)
    )

    assert loaded_metadata["n_features"] == (
        loaded_matrix.shape[1]
    )

    np.testing.assert_array_equal(
        loaded_product_ids.to_numpy(),
        content_product_ids.to_numpy(),
    )

    np.testing.assert_array_equal(
        loaded_catalog["PRODUCT_ID"].to_numpy(),
        loaded_product_ids.to_numpy(),
    )

    # 18. Validate loaded preprocessing pipeline
    restored_matrix = (
        loaded_preprocessor
        .transform(loaded_catalog)
        .tocsr()
    )

    assert restored_matrix.shape == loaded_matrix.shape

    np.testing.assert_allclose(
        restored_matrix.toarray()[0],
        loaded_matrix.toarray()[0],
        rtol=1e-6,
        atol=1e-6,
    )

    # 19. Generate predictions using loaded model
    loaded_predictions = recommend_content_based(
        product_id=EXAMPLE_PRODUCT_ID,
        model=loaded_model,
        content_matrix=loaded_matrix,
        product_ids=loaded_product_ids,
        content_catalog=loaded_catalog,
        top_n=TOP_N,
    )

    # 20. Compare original and loaded predictions
    pd.testing.assert_frame_equal(
        original_predictions,
        loaded_predictions,
        check_dtype=False,
        rtol=1e-6,
        atol=1e-6,
    )

    # 21. Validate recommended products
    assert EXAMPLE_PRODUCT_ID not in (
        loaded_predictions["PRODUCT_ID"].values
    )

    assert loaded_predictions["PRODUCT_ID"].is_unique

    assert loaded_predictions[
        "cosine_similarity"
    ].between(0, 1.000001).all()

    print("\nSAVED MODEL VALIDATION")
    print("Preprocessor Reload: PASSED")
    print("Model Reload: PASSED")
    print("Matrix Reload: PASSED")
    print("Prediction Comparison: PASSED")

    print("\nPRODUCT RECOMMENDATIONS")
    print(
        loaded_predictions.round(4).to_string(
            index=False
        )
    )

    print(
        "\nCONTENT-BASED TRAINING PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
