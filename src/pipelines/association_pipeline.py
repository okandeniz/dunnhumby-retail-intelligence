
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data.database import get_engine

from src.data.recommendations import (
    load_basket_product_interactions,
    load_product_catalog,
    prepare_valid_catalog,
)

from src.features.recommendations import (
    build_product_frequency,
)

from src.recommendations.association_rules import (
    build_basket_product_matrix,
    mine_frequent_itemsets,
    generate_association_rules,
    build_recommendation_rules,
    recommend_products,
    recommend_association_customer,
)


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "association_rules"
)

TRAIN_END_DAY = 621

MIN_SUPPORT = 0.002
MIN_CONFIDENCE = 0.10
MIN_LIFT = 1.20
MAX_ITEMSET_LENGTH = 2

EXAMPLE_PRODUCT_ID = 826249
EXAMPLE_CUSTOMER_ID = 1


def main() -> None:
    """Train, save and validate Association Rules."""

    print("ASSOCIATION RULES TRAINING PIPELINE")
    print("-" * 40)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Load training data
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

    # 2. Prepare valid product catalog
    valid_catalog = prepare_valid_catalog(
        product_catalog=product_catalog,
    )

    train_items = train_items.loc[
        train_items["PRODUCT_ID"].isin(
            valid_catalog["PRODUCT_ID"]
        )
    ].copy()

    # 3. Validate training data
    assert len(train_items) == 2_178_447

    n_baskets = train_items["BASKET_ID"].nunique()

    assert n_baskets == 217_341

    print("\nDATA PREPARATION COMPLETED")
    print("Training Interactions:", len(train_items))
    print("Training Baskets:", n_baskets)

    # 4. Calculate product frequency
    product_frequency = build_product_frequency(
        train_items=train_items,
        valid_catalog=valid_catalog,
    )

    # 5. Build sparse basket-product matrix
    basket_encoded, basket_matrix = (
        build_basket_product_matrix(
            train_items=train_items,
            product_frequency=product_frequency,
            min_support=MIN_SUPPORT,
        )
    )

    assert basket_matrix.shape == (217_341, 575)
    assert basket_matrix.nnz == 630_747

    print("\nBASKET-PRODUCT MATRIX COMPLETED")
    print("Matrix Shape:", basket_matrix.shape)
    print("Non-Zero Interactions:", basket_matrix.nnz)

    # 6. Run FP-Growth
    frequent_itemsets = mine_frequent_itemsets(
        basket_encoded=basket_encoded,
        min_support=MIN_SUPPORT,
        max_len=MAX_ITEMSET_LENGTH,
    )

    itemset_counts = (
        frequent_itemsets["itemset_size"]
        .value_counts()
    )

    assert itemset_counts.loc[1] == 575
    assert itemset_counts.loc[2] == 136

    print("\nFP-GROWTH COMPLETED")
    print("Frequent Itemsets:", len(frequent_itemsets))

    # 7. Generate Association Rules
    rules, association_results = (
        generate_association_rules(
            frequent_itemsets=frequent_itemsets,
            valid_catalog=valid_catalog,
            n_baskets=n_baskets,
            min_support=MIN_SUPPORT,
            min_confidence=MIN_CONFIDENCE,
            min_lift=MIN_LIFT,
        )
    )

    assert len(rules) == 146

    # 8. Prepare product recommendation rules
    recommendation_rules = build_recommendation_rules(
        rules=rules,
        valid_catalog=valid_catalog,
    )

    assert len(recommendation_rules) == 146

    print("\nASSOCIATION RULE GENERATION COMPLETED")
    print("Total Rules:", len(recommendation_rules))

    # 9. Build customer purchase histories
    train_purchase_sets = (
        train_items
        .groupby("household_key")["PRODUCT_ID"]
        .agg(set)
        .to_dict()
    )

    print("Customers With Purchase History:", len(train_purchase_sets))

    # 10. Generate product recommendations
    original_product_predictions = recommend_products(
        product_id=EXAMPLE_PRODUCT_ID,
        recommendation_rules=recommendation_rules,
        top_n=5,
    )

    assert len(original_product_predictions) == 4
    assert original_product_predictions["consequent_id"].is_unique

    # 11. Generate customer recommendations
    original_customer_predictions = (
        recommend_association_customer(
            household_key=EXAMPLE_CUSTOMER_ID,
            evaluation_rules=recommendation_rules,
            train_purchase_sets=train_purchase_sets,
            top_n=10,
        )
    )

    # 12. Define artifact paths
    rules_path = MODEL_DIR / "recommendation_rules.joblib"
    purchases_path = MODEL_DIR / "train_purchase_sets.joblib"
    readable_path = MODEL_DIR / "association_rules.csv"
    itemsets_path = MODEL_DIR / "frequent_itemsets.csv"
    metadata_path = MODEL_DIR / "association_metadata.json"

    # 13. Save recommendation rules
    joblib.dump(
        recommendation_rules,
        rules_path,
    )

    # 14. Save historical purchase sets
    joblib.dump(
        train_purchase_sets,
        purchases_path,
    )

    # 15. Save readable Association Rules
    association_results.to_csv(
        readable_path,
        index=False,
    )

    # 16. Save readable frequent itemsets
    itemsets_export = frequent_itemsets[
        ["support", "itemsets", "itemset_size"]
    ].copy()

    itemsets_export["product_ids"] = (
        itemsets_export["itemsets"].apply(
            lambda items: "|".join(
                sorted(map(str, items), key=int)
            )
        )
    )

    itemsets_export[
        ["product_ids", "itemset_size", "support"]
    ].to_csv(
        itemsets_path,
        index=False,
    )

    # 17. Save model metadata
    metadata = {
        "model_name": "FP-Growth Association Rules",
        "training_end_day": TRAIN_END_DAY,
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "min_lift": MIN_LIFT,
        "max_itemset_length": MAX_ITEMSET_LENGTH,
        "training_interactions": int(len(train_items)),
        "training_baskets": int(n_baskets),
        "frequent_products": 575,
        "frequent_itemsets": int(len(frequent_itemsets)),
        "association_rules": int(len(recommendation_rules)),
        "customers": int(len(train_purchase_sets)),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nMODEL ARTIFACTS SAVED")
    print("Output Directory:", MODEL_DIR)

    # 18. Reload saved artifacts
    loaded_rules = joblib.load(
        rules_path
    )

    loaded_purchase_sets = joblib.load(
        purchases_path
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        loaded_metadata = json.load(file)

    # 19. Validate loaded artifacts
    pd.testing.assert_frame_equal(
        recommendation_rules,
        loaded_rules,
    )

    assert train_purchase_sets == loaded_purchase_sets
    assert loaded_metadata["association_rules"] == 146

    # 20. Generate product recommendations again
    loaded_product_predictions = recommend_products(
        product_id=EXAMPLE_PRODUCT_ID,
        recommendation_rules=loaded_rules,
        top_n=5,
    )

    pd.testing.assert_frame_equal(
        original_product_predictions,
        loaded_product_predictions,
    )

    # 21. Generate customer recommendations again
    loaded_customer_predictions = (
        recommend_association_customer(
            household_key=EXAMPLE_CUSTOMER_ID,
            evaluation_rules=loaded_rules,
            train_purchase_sets=loaded_purchase_sets,
            top_n=10,
        )
    )

    pd.testing.assert_frame_equal(
        original_customer_predictions,
        loaded_customer_predictions,
    )

    print("\nSAVED ARTIFACT VALIDATION")
    print("Association Rules Reload: PASSED")
    print("Customer History Reload: PASSED")
    print("Product Recommendation Comparison: PASSED")
    print("Customer Recommendation Comparison: PASSED")

    print("\nPRODUCT RECOMMENDATIONS")
    print("Selected Product ID:", EXAMPLE_PRODUCT_ID)
    print(
        loaded_product_predictions.round(4).to_string(
            index=False
        )
    )

    print("\nCUSTOMER RECOMMENDATIONS")
    print("Selected Customer ID:", EXAMPLE_CUSTOMER_ID)
    print(
        loaded_customer_predictions.round(4).to_string(
            index=False
        )
    )

    print(
        "\nASSOCIATION RULES TRAINING PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
