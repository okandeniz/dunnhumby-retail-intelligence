
import json

from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from src.data.database import get_engine

from src.data.recommendations import (
    load_basket_product_interactions,
    load_product_catalog,
    prepare_valid_catalog,
    split_recommendation_data,
)

from src.evaluation.recommendations import (
    prepare_evaluation_data,
    evaluate_recommender,
)

from src.recommendations.user_based import (
    recommend_user_based,
)

from src.recommendations.matrix_factorization import (
    recommend_svd,
    recommend_nmf,
)

from src.features.recommendations import (
    build_product_frequency,
)

from src.recommendations.association_rules import (
    build_basket_product_matrix,
    mine_frequent_itemsets,
    generate_association_rules,
    build_recommendation_rules,
    recommend_association_customer,
)

from src.recommendations.item_based import (
    build_customer_seed_products,
    create_item_neighbor_lookup,
    recommend_item_based_customer,
)

from src.recommendations.content_based import (
    create_content_neighbor_lookup,
    recommend_content_customer,
)

import warnings

warnings.filterwarnings(
    "ignore",
    message=(
        r".*sklearn\.utils\.parallel\.delayed.*"
        r"sklearn\.utils\.parallel\.Parallel.*"
    ),
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)



PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "recommendation"

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "recommendation"
)

TRAIN_END_DAY = 621
TEST_END_DAY = 711
TOP_K = 10


def load_factor_model(
    model_name: str,
) -> dict:
    """Load saved SVD or NMF artifacts without retraining."""

    model_dir = MODEL_DIR / model_name

    # 1. Load trained model and metadata
    model = joblib.load(
        model_dir / f"{model_name}_model.joblib"
    )

    metadata = joblib.load(
        model_dir / f"{model_name}_metadata.joblib"
    )

    # 2. Load latent factors
    with np.load(
        model_dir / f"{model_name}_factors.npz",
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

    # 3. Load historical interactions
    customer_item_matrix = load_npz(
        model_dir / "customer_item_matrix.npz"
    ).tocsr()

    # 4. Load product descriptions
    catalog = pd.read_csv(
        model_dir / "product_catalog.csv"
    )

    # 5. Validate artifacts
    assert customer_item_matrix.shape == (
        len(customer_ids),
        len(product_ids),
    )

    assert user_factors.shape == (
        len(customer_ids),
        metadata["n_components"],
    )

    assert item_factors.shape == (
        len(product_ids),
        metadata["n_components"],
    )

    np.testing.assert_allclose(
        item_factors,
        model.components_.T,
        rtol=1e-6,
        atol=1e-6,
    )

    if set(catalog["PRODUCT_ID"]) != set(product_ids):
        raise ValueError("Product catalog mismatch.")

    return {
        "model": model,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "customer_item_matrix": customer_item_matrix,
        "customer_ids": customer_ids,
        "product_ids": product_ids,
        "catalog": catalog,
        "metadata": metadata,
    }


def main() -> None:
    """Evaluate saved recommendation models on future purchases."""

    print("RECOMMENDATION EVALUATION PIPELINE")
    print("-" * 40)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load historical and future interactions
    engine = get_engine()

    try:
        interactions = load_basket_product_interactions(
            engine=engine,
            end_day=TEST_END_DAY,
        )

        product_catalog = load_product_catalog(
            engine=engine,
        )
    finally:
        engine.dispose()

    # 2. Prepare valid retail products
    valid_catalog = prepare_valid_catalog(
        product_catalog=product_catalog,
    )

    interactions = interactions.loc[
        interactions["PRODUCT_ID"].isin(
            valid_catalog["PRODUCT_ID"]
        )
    ].copy()

    # 3. Apply chronological split
    train_items, test_items = split_recommendation_data(
        interactions=interactions,
        train_end_day=TRAIN_END_DAY,
        test_end_day=TEST_END_DAY,
    )

    print("\nDATA LOADING COMPLETED")
    print("Training Interactions:", len(train_items))
    print("Test Interactions:", len(test_items))

    # 4. Load User-Based KNN artifacts
    user_dir = MODEL_DIR / "user_based"

    user_model = joblib.load(
        user_dir / "user_knn.joblib"
    )

    user_matrix = load_npz(
        user_dir / "customer_item_matrix.npz"
    ).tocsr()

    user_metadata = joblib.load(
        user_dir / "user_based_metadata.joblib"
    )

    user_catalog = pd.read_csv(
        user_dir / "product_catalog.csv"
    )

    customer_ids = pd.Index(
        user_metadata["customer_ids"]
    )

    product_ids = pd.Index(
        user_metadata["product_ids"]
    )

    assert user_matrix.shape == (
        len(customer_ids),
        len(product_ids),
    )

    assert user_model.n_features_in_ == len(product_ids)

    if set(user_catalog["PRODUCT_ID"]) != set(product_ids):
        raise ValueError("User-Based catalog mismatch.")

    # 5. Load matrix factorization models
    svd_artifacts = load_factor_model("svd")
    nmf_artifacts = load_factor_model("nmf")

    # 6. Validate common customer and product indexes
    for name, artifacts in [
        ("SVD", svd_artifacts),
        ("NMF", nmf_artifacts),
    ]:
        np.testing.assert_array_equal(
            customer_ids.to_numpy(),
            artifacts["customer_ids"].to_numpy(),
        )

        np.testing.assert_array_equal(
            product_ids.to_numpy(),
            artifacts["product_ids"].to_numpy(),
        )

        if (
            user_matrix
            != artifacts["customer_item_matrix"]
        ).nnz != 0:
            raise ValueError(
                f"{name} interaction matrix mismatch."
            )

    print("\nSAVED MODEL LOADING: PASSED")

    # 7. Prepare future new-product evaluation data
    (
        novel_test_items,
        actual_products,
        evaluation_customers,
    ) = prepare_evaluation_data(
        train_items=train_items,
        test_items=test_items,
        customer_ids=customer_ids,
        product_ids=product_ids,
    )

    assert len(novel_test_items) == 109658
    assert len(evaluation_customers) == 2266

    print("\nEVALUATION DATA PREPARATION: PASSED")
    print("Novel Test Pairs:", len(novel_test_items))
    print("Evaluation Customers:", len(evaluation_customers))
    print("Catalog Size:", len(product_ids))

    # 8. Build popularity ranking from training interactions
    product_counts = pd.Series(
        np.asarray(
            user_matrix.getnnz(axis=0)
        ).ravel(),
        index=product_ids,
    )

    popular_product_ids = (
        product_counts
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # 9. Define popularity recommendation function
    def recommend_popular(
        household_key: int,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Recommend popular products not purchased before."""

        if top_n < 1:
            raise ValueError("top_n must be positive.")

        customer_idx = customer_ids.get_indexer(
            [household_key]
        )[0]

        if customer_idx == -1:
            raise ValueError("Unknown customer.")

        purchased_ids = set(
            product_ids.take(
                user_matrix[customer_idx].indices
            )
        )

        recommendations = []

        for product_id in popular_product_ids:
            if product_id not in purchased_ids:
                recommendations.append(product_id)

            if len(recommendations) == top_n:
                break

        return pd.DataFrame({
            "PRODUCT_ID": recommendations
        })

    # Prepare Association Rules using training data only
    print("\nPREPARING ASSOCIATION RULES")

    # 1. Calculate product support
    product_frequency = build_product_frequency(
        train_items=train_items,
        valid_catalog=valid_catalog,
    )

    # 2. Build sparse basket-product matrix
    basket_encoded, _ = build_basket_product_matrix(
        train_items=train_items,
        product_frequency=product_frequency,
        min_support=0.002,
    )

    # 3. Mine frequent itemsets
    frequent_itemsets = mine_frequent_itemsets(
        basket_encoded=basket_encoded,
        min_support=0.002,
        max_len=2,
    )

    # 4. Generate association rules
    rules, _ = generate_association_rules(
        frequent_itemsets=frequent_itemsets,
        valid_catalog=valid_catalog,
        n_baskets=len(basket_encoded),
        min_support=0.002,
        min_confidence=0.10,
        min_lift=1.20,
    )

    # 5. Prepare recommendation rules
    recommendation_rules = build_recommendation_rules(
        rules=rules,
        valid_catalog=valid_catalog,
    )

    assert len(recommendation_rules) == 146

    # 6. Keep products supported by evaluation catalog
    evaluation_rules = recommendation_rules.loc[
        recommendation_rules["consequent_id"].isin(
            product_ids
        )
    ].copy()

    # 7. Build customer purchase sets
    train_purchase_sets = (
        train_items
        .groupby("household_key")["PRODUCT_ID"]
        .agg(set)
        .to_dict()
    )

    print("Association Rules:", len(recommendation_rules))
    print("Evaluation Rules:", len(evaluation_rules))
    print("Association Rules Preparation: PASSED")

    # Prepare Item-Based KNN evaluation
    print("\nPREPARING ITEM-BASED KNN")

    item_dir = MODEL_DIR / "item_based"

    # 1. Load saved Item-Based KNN model
    item_model = joblib.load(
        item_dir / "item_knn.joblib"
    )

    # 2. Load sparse item-user matrix
    item_user_matrix = load_npz(
        item_dir / "item_user_matrix.npz"
    ).tocsr()

    # 3. Load model metadata
    item_metadata = joblib.load(
        item_dir / "item_based_metadata.joblib"
    )

    item_product_ids = pd.Index(
        item_metadata["product_ids"]
    )

    # 4. Validate product indexes
    np.testing.assert_array_equal(
        item_product_ids.to_numpy(),
        product_ids.to_numpy(),
    )

    assert item_user_matrix.shape == (
        len(product_ids),
        len(customer_ids),
    )

    assert (
        item_user_matrix != user_matrix.T.tocsr()
    ).nnz == 0

    assert item_model.n_features_in_ == len(customer_ids)

    print("Saved Item-Based Model: PASSED")

    # 5. Prepare recent customer purchases
    customer_seed_products = build_customer_seed_products(
        train_items=train_items,
        product_ids=product_ids,
        n_seed_items=10,
    )

    assert len(customer_seed_products) == 2481

    # 6. Validate notebook customer example
    expected_seeds = [
        829323,
        851515,
        879280,
        940947,
        968164,
        979792,
        995242,
        1055201,
        1069575,
        1074612,
    ]

    assert customer_seed_products[1] == expected_seeds

    print("Customer Seed Preparation: PASSED")

    # 7. Create cached item-neighbor lookup
    item_model.set_params(n_jobs=1)
    item_neighbor_lookup = create_item_neighbor_lookup(
        model=item_model,
        item_user_matrix=item_user_matrix,
        product_ids=product_ids,
        n_item_neighbors=30,
    )

    # 8. Test customer-level recommendations
    example_predictions = recommend_item_based_customer(
        household_key=1,
        customer_item_matrix=user_matrix,
        customer_ids=customer_ids,
        product_ids=product_ids,
        customer_seed_products=customer_seed_products,
        get_item_neighbors=item_neighbor_lookup,
        top_n=10,
    )

    expected_top5 = [
        1127831,
        981760,
        866211,
        878996,
        854852,
    ]

    np.testing.assert_array_equal(
        example_predictions["PRODUCT_ID"]
        .head(5)
        .to_numpy(),
        expected_top5,
    )

    print("Item-Based Recommendation Validation: PASSED")
    print("Customer 1 Seed Products:", customer_seed_products[1])
    print("\nCustomer 1 Recommendations")
    print(example_predictions.head(10).round(4).to_string(index=False))

    # Prepare Content-Based Filtering evaluation
    print("\nPREPARING CONTENT-BASED FILTERING")

    content_dir = MODEL_DIR / "content_based"

    # 1. Load saved Content-Based KNN
    content_model = joblib.load(
        content_dir / "content_knn.joblib"
    )

    # Use one worker for neighbor queries
    content_model.set_params(n_jobs=1)

    # 2. Load sparse content matrix
    content_matrix = load_npz(
        content_dir / "content_matrix.npz"
    ).tocsr()

    # 3. Load content metadata
    content_metadata = joblib.load(
        content_dir / "content_based_metadata.joblib"
    )

    content_product_ids = pd.Index(
        content_metadata["product_ids"]
    )

    # 4. Load product catalog
    content_catalog = pd.read_csv(
        content_dir / "content_product_catalog.csv",
        keep_default_na=False,
    )

    # 5. Validate common product indexes
    np.testing.assert_array_equal(
        content_product_ids.to_numpy(),
        product_ids.to_numpy(),
    )

    np.testing.assert_array_equal(
        content_catalog["PRODUCT_ID"].to_numpy(),
        content_product_ids.to_numpy(),
    )

    # 6. Validate model dimensions
    assert content_matrix.shape == (
        content_metadata["n_products"],
        content_metadata["n_features"],
    )

    assert content_model.n_features_in_ == (
        content_matrix.shape[1]
    )

    assert content_matrix.shape == (11603, 4712)

    print("Saved Content-Based Model: PASSED")
    print("Content Matrix:", content_matrix.shape)

    # 7. Create cached content-neighbor lookup
    content_neighbor_lookup = (
        create_content_neighbor_lookup(
            model=content_model,
            content_matrix=content_matrix,
            product_ids=content_product_ids,
            n_neighbors=30,
        )
    )

    # 8. Test customer-level content recommendations
    content_example = recommend_content_customer(
        household_key=1,
        customer_item_matrix=user_matrix,
        customer_ids=customer_ids,
        product_ids=product_ids,
        customer_seed_products=customer_seed_products,
        get_content_neighbors=content_neighbor_lookup,
        top_n=10,
    )

    # 9. Validate example recommendations
    assert len(content_example) == 10

    assert content_example["PRODUCT_ID"].is_unique

    customer_idx = customer_ids.get_loc(1)

    purchased_ids = set(
        product_ids.take(
            user_matrix[customer_idx].indices
        )
    )

    assert not set(
        content_example["PRODUCT_ID"]
    ).intersection(purchased_ids)

    print("Content-Based Recommendation Validation: PASSED")

    print("\nCUSTOMER 1 CONTENT RECOMMENDATIONS")
    print(
        content_example.round(4).to_string(index=False)
    )

    # 10. Prepare reusable recommendation functions
    recommenders = {
        "Popularity Baseline": recommend_popular,

        "User-Based CF": partial(
            recommend_user_based,
            model=user_model,
            customer_item_matrix=user_matrix,
            customer_ids=customer_ids,
            product_ids=product_ids,
            valid_catalog=user_catalog,
            n_neighbors=30,
        ),

        "Truncated SVD": partial(
            recommend_svd,
            user_factors=svd_artifacts["user_factors"],
            item_factors=svd_artifacts["item_factors"],
            customer_item_matrix=svd_artifacts[
                "customer_item_matrix"
            ],
            customer_ids=svd_artifacts["customer_ids"],
            product_ids=svd_artifacts["product_ids"],
            valid_catalog=svd_artifacts["catalog"],
        ),

        "NMF": partial(
            recommend_nmf,
            user_factors=nmf_artifacts["user_factors"],
            item_factors=nmf_artifacts["item_factors"],
            customer_item_matrix=nmf_artifacts[
                "customer_item_matrix"
            ],
            customer_ids=nmf_artifacts["customer_ids"],
            product_ids=nmf_artifacts["product_ids"],
            valid_catalog=nmf_artifacts["catalog"],
        ),

        "Association Rules": partial(
            recommend_association_customer,
            evaluation_rules=evaluation_rules,
            train_purchase_sets=train_purchase_sets,
        ),

        "Item-Based CF": partial(
            recommend_item_based_customer,
            customer_item_matrix=user_matrix,
            customer_ids=customer_ids,
            product_ids=product_ids,
            customer_seed_products=customer_seed_products,
            get_item_neighbors=item_neighbor_lookup,
        ),

        "Content-Based Filtering": partial(
            recommend_content_customer,
            customer_item_matrix=user_matrix,
            customer_ids=customer_ids,
            product_ids=product_ids,
            customer_seed_products=customer_seed_products,
            get_content_neighbors=content_neighbor_lookup,
        ),
    }

    # 11. Evaluate each model
    model_results = []

    for model_name, recommend_fn in recommenders.items():
        print(f"\nEVALUATING: {model_name}")

        summary, details = evaluate_recommender(
            recommend_fn=recommend_fn,
            customers=evaluation_customers,
            actual_products=actual_products,
            catalog_size=len(product_ids),
            k=TOP_K,
        )

        model_results.append({
            "model": model_name,
            **summary,
        })

        # Save customer-level evaluation results
        filename = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + "_evaluation.csv"
        )

        details.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

        print("Evaluation Customers:", len(details))
        print(
            "Precision@10:",
            round(summary["precision_at_k"], 4),
        )
        print(
            "Hit Rate@10:",
            round(summary["hit_rate_at_k"], 4),
        )

    print("\nITEM-BASED NEIGHBOR CACHE")
    print(item_neighbor_lookup.cache_info())

    # 12. Create model comparison table
    comparison_df = (
        pd.DataFrame(model_results)
        .set_index("model")
        .sort_values(
            "precision_at_k",
            ascending=False,
        )
    )

    print("\nRECOMMENDATION MODEL COMPARISON")
    print(comparison_df.round(4).to_string())

    # 13. Save model comparison
    comparison_df.to_csv(
        OUTPUT_DIR / "model_comparison_stage4.csv"
    )

    # 14. Save evaluation dataset
    novel_test_items.to_csv(
        OUTPUT_DIR / "novel_test_items.csv",
        index=False,
    )

    # 15. Save evaluation metadata
    metadata = {
        "evaluation_name": "Recommendation Model Comparison",
        "train_end_day": TRAIN_END_DAY,
        "test_end_day": TEST_END_DAY,
        "top_k": TOP_K,
        "evaluation_customers": len(evaluation_customers),
        "novel_test_pairs": len(novel_test_items),
        "catalog_size": len(product_ids),
        "models": list(recommenders),
        "nmf_converged": bool(
            nmf_artifacts["metadata"]["converged"]
        ),
        "evaluation_type": "exploratory",
    }

    with open(
        OUTPUT_DIR / "evaluation_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    # 16. Validate saved model comparison
    saved_comparison = pd.read_csv(
        OUTPUT_DIR / "model_comparison_stage4.csv",
        index_col="model",
    )

    pd.testing.assert_frame_equal(
        comparison_df,
        saved_comparison,
        check_dtype=False,
        rtol=1e-9,
        atol=1e-9,
    )

    print("\nEVALUATION ARTIFACTS: PASSED")

    print(
        "\nRECOMMENDATION EVALUATION PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
