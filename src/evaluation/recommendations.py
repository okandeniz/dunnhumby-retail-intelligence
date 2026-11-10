
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd


def prepare_evaluation_data(
    train_items: pd.DataFrame,
    test_items: pd.DataFrame,
    customer_ids: pd.Index,
    product_ids: pd.Index,
) -> tuple[pd.DataFrame, dict, pd.Index]:
    """Prepare future new-product purchases for evaluation."""

    required = {"household_key", "PRODUCT_ID"}

    # 1. Validate input columns
    for name, data in [
        ("train_items", train_items),
        ("test_items", test_items),
    ]:
        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {name}: {sorted(missing)}"
            )

    if train_items.empty or test_items.empty:
        raise ValueError("Training or test data is empty.")

    if not customer_ids.is_unique or not product_ids.is_unique:
        raise ValueError("Duplicate customer or product IDs.")

    # 2. Check chronological separation
    if (
        "basket_day" in train_items.columns
        and "basket_day" in test_items.columns
    ):
        if train_items["basket_day"].max() >= (
            test_items["basket_day"].min()
        ):
            raise ValueError("Training and test periods overlap.")

    # 3. Get unique future purchases
    test_pairs = (
        test_items[
            ["household_key", "PRODUCT_ID"]
        ]
        .drop_duplicates()
    )

    # 4. Keep known customers and eligible products
    eligible_test = test_pairs.loc[
        test_pairs["household_key"].isin(customer_ids)
        & test_pairs["PRODUCT_ID"].isin(product_ids)
    ].copy()

    # 5. Get historical eligible purchases
    train_pairs = (
        train_items.loc[
            train_items["household_key"].isin(customer_ids)
            & train_items["PRODUCT_ID"].isin(product_ids),
            ["household_key", "PRODUCT_ID"],
        ]
        .drop_duplicates()
    )

    # 6. Mark previously purchased products
    evaluation_pairs = eligible_test.merge(
        train_pairs.assign(purchased_before=1),
        on=["household_key", "PRODUCT_ID"],
        how="left",
        validate="one_to_one",
    )

    # 7. Keep only new future purchases
    novel_test_items = evaluation_pairs.loc[
        evaluation_pairs["purchased_before"].isna(),
        ["household_key", "PRODUCT_ID"],
    ].copy()

    # 8. Create actual product sets
    actual_products = (
        novel_test_items
        .groupby("household_key")["PRODUCT_ID"]
        .agg(set)
        .to_dict()
    )

    # 9. Select customers with future new products
    evaluation_customers = pd.Index(
        sorted(actual_products.keys())
    )

    if len(evaluation_customers) == 0:
        raise ValueError("No eligible evaluation customers.")

    return (
        novel_test_items,
        actual_products,
        evaluation_customers,
    )


def evaluate_recommender(
    recommend_fn: Callable[..., pd.DataFrame],
    customers: pd.Index,
    actual_products: Mapping[int, set],
    catalog_size: int,
    k: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate top-K recommendations against future purchases."""

    # 1. Validate parameters
    if k < 1:
        raise ValueError("k must be positive.")

    if catalog_size < 1:
        raise ValueError("catalog_size must be positive.")

    if len(customers) == 0:
        raise ValueError("Evaluation customers are empty.")

    if not customers.is_unique:
        raise ValueError("Duplicate evaluation customers.")

    results = []
    all_recommended = set()

    # 2. Evaluate each customer
    for household_key in customers:
        household_key = int(household_key)

        if household_key not in actual_products:
            raise ValueError(
                f"Missing actual purchases for {household_key}."
            )

        # Generate top-K recommendations
        recommendations = recommend_fn(
            household_key=household_key,
            top_n=k,
        )

        if "PRODUCT_ID" not in recommendations.columns:
            raise ValueError("Recommendations need PRODUCT_ID.")

        recommended_ids = (
            recommendations["PRODUCT_ID"]
            .drop_duplicates()
            .head(k)
            .tolist()
        )

        # Get actual future new purchases
        actual_ids = set(
            actual_products[household_key]
        )

        if not actual_ids:
            raise ValueError("Actual purchase set is empty.")

        # 3. Find correct recommendations
        hits = len(
            set(recommended_ids).intersection(actual_ids)
        )

        # 4. Calculate customer-level metrics
        precision = hits / k
        recall = hits / len(actual_ids)
        hit_rate = int(hits > 0)

        all_recommended.update(recommended_ids)

        results.append({
            "household_key": household_key,
            "precision": precision,
            "recall": recall,
            "hit_rate": hit_rate,
        })

    # 5. Calculate average metrics
    results_df = pd.DataFrame(results)

    coverage = len(all_recommended) / catalog_size

    if coverage > 1:
        raise ValueError("Catalog coverage exceeds 100%.")

    summary = {
        "precision_at_k": float(
            results_df["precision"].mean()
        ),
        "recall_at_k": float(
            results_df["recall"].mean()
        ),
        "hit_rate_at_k": float(
            results_df["hit_rate"].mean()
        ),
        "coverage_at_k": float(coverage),
    }

    # 6. Validate evaluation metrics
    if not all(
        np.isfinite(value) and 0 <= value <= 1
        for value in summary.values()
    ):
        raise ValueError("Invalid evaluation metrics.")

    return summary, results_df
