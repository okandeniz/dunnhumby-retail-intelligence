
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.recommendations.association_rules import (
    recommend_products,
    recommend_association_customer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "association_rules"
)


def run_association_inference(
    product_id: int | None = None,
    household_key: int | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Generate recommendations using saved association rules."""

    # 1. Validate inference mode
    if (product_id is None) == (household_key is None):
        raise ValueError(
            "Provide either product_id or household_key."
        )

    if top_n is None:
        top_n = 5 if product_id is not None else 10

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    # 2. Load saved recommendation rules
    rules_path = MODEL_DIR / "recommendation_rules.joblib"

    recommendation_rules = joblib.load(
        rules_path
    )

    # 3. Load model metadata
    with open(
        MODEL_DIR / "association_metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    # 4. Validate saved rules
    if len(recommendation_rules) != metadata["association_rules"]:
        raise ValueError("Association rule count mismatch.")

    required = {
        "antecedent_id",
        "consequent_id",
        "support",
        "confidence",
        "lift",
    }

    if not required.issubset(recommendation_rules.columns):
        raise ValueError("Missing association rule columns.")

    # 5. Product-Based Recommendation
    if product_id is not None:
        if product_id < 1:
            raise ValueError("product_id must be positive.")

        recommendations = recommend_products(
            product_id=product_id,
            recommendation_rules=recommendation_rules,
            top_n=top_n,
        )

        if not recommendations["consequent_id"].is_unique:
            raise ValueError("Duplicate product recommendations.")

        if product_id in recommendations["consequent_id"].values:
            raise ValueError("Selected product was recommended.")

        return recommendations

    # 6. Load historical customer purchases
    train_purchase_sets = joblib.load(
        MODEL_DIR / "train_purchase_sets.joblib"
    )

    if len(train_purchase_sets) != metadata["customers"]:
        raise ValueError("Customer history count mismatch.")

    # 7. Customer-Based Recommendation
    recommendations = recommend_association_customer(
        household_key=household_key,
        evaluation_rules=recommendation_rules,
        train_purchase_sets=train_purchase_sets,
        top_n=top_n,
    )

    # 8. Validate customer recommendations
    if not recommendations["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate product recommendations.")

    purchased_ids = train_purchase_sets[household_key]

    if set(recommendations["PRODUCT_ID"]) & purchased_ids:
        raise ValueError(
            "Previously purchased product was recommended."
        )

    if len(recommendations) > top_n:
        raise ValueError("Too many recommendations.")

    scores = recommendations[
        "recommendation_score"
    ].to_numpy()

    if not np.isfinite(scores).all():
        raise ValueError("Invalid recommendation scores.")

    return recommendations


def main() -> None:
    """Run Association Rules inference without retraining."""

    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Association Rules Product Recommendation"
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--product-id",
        type=int,
    )

    mode.add_argument(
        "--household-key",
        type=int,
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    print("ASSOCIATION RULES INFERENCE PIPELINE")
    print("-" * 40)

    # 2. Generate recommendations
    recommendations = run_association_inference(
        product_id=args.product_id,
        household_key=args.household_key,
        top_n=args.top_n,
    )

    print("\nINFERENCE COMPLETED")

    if args.product_id is not None:
        print("Inference Type: Product-Based")
        print("Selected Product ID:", args.product_id)
    else:
        print("Inference Type: Customer-Based")
        print("Selected Customer ID:", args.household_key)

    print("Recommendations:", len(recommendations))

    print("\nPRODUCT RECOMMENDATIONS")

    if recommendations.empty:
        print("No matching association rules found.")
    else:
        print(
            recommendations.round(4).to_string(
                index=False
            )
        )

    # 3. Validate notebook product example
    if args.product_id == 826249:
        expected_ids = [
            1098066,
            1082185,
            1029743,
            883404,
        ]

        if args.top_n is None or args.top_n >= 4:
            np.testing.assert_array_equal(
                recommendations["consequent_id"]
                .head(4)
                .to_numpy(),
                expected_ids,
            )

            print("\nProduct Reference Comparison: PASSED")

    # 4. Validate notebook customer example
    if args.household_key == 1:
        expected_ids = [
            8090537,
            908531,
            854852,
            834484,
            981760,
        ]

        if args.top_n is None or args.top_n >= 5:
            np.testing.assert_array_equal(
                recommendations["PRODUCT_ID"]
                .head(5)
                .to_numpy(),
                expected_ids,
            )

            print("\nCustomer Reference Comparison: PASSED")

    print("\nSaved Rules Inference: PASSED")

    print(
        "\nASSOCIATION RULES INFERENCE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
