
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from collections.abc import Mapping


def build_basket_product_matrix(
    train_items: pd.DataFrame,
    product_frequency: pd.DataFrame,
    min_support: float = 0.002,
) -> tuple[pd.DataFrame, csr_matrix]:
    """Build a sparse basket-product matrix for FP-Growth."""

    # 1. Validate inputs
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be between 0 and 1.")

    required_items = {"BASKET_ID", "PRODUCT_ID"}
    required_frequency = {"PRODUCT_ID", "support"}

    missing_items = required_items - set(train_items.columns)
    missing_frequency = (
        required_frequency - set(product_frequency.columns)
    )

    if missing_items or missing_frequency:
        raise ValueError(
            f"Missing columns: {sorted(missing_items | missing_frequency)}"
        )

    if train_items.empty or product_frequency.empty:
        raise ValueError("Input data must not be empty.")

    if product_frequency["PRODUCT_ID"].duplicated().any():
        raise ValueError("Duplicate product frequency records.")

    # 2. Select frequent products
    frequent_product_ids = pd.Index(
        product_frequency.loc[
            product_frequency["support"] >= min_support,
            "PRODUCT_ID",
        ].unique()
    )

    if len(frequent_product_ids) == 0:
        raise ValueError("No products meet the minimum support.")

    # 3. Keep eligible basket-product relationships
    filtered_items = (
        train_items.loc[
            train_items["PRODUCT_ID"].isin(frequent_product_ids),
            ["BASKET_ID", "PRODUCT_ID"],
        ]
        .drop_duplicates()
    )

    # 4. Keep ALL training baskets
    basket_ids = pd.Index(
        train_items["BASKET_ID"].unique()
    )

    # 5. Convert IDs into matrix positions
    row_indices = basket_ids.get_indexer(
        filtered_items["BASKET_ID"]
    )

    col_indices = frequent_product_ids.get_indexer(
        filtered_items["PRODUCT_ID"]
    )

    if (row_indices < 0).any() or (col_indices < 0).any():
        raise ValueError("Invalid basket or product index.")

    # 6. Create sparse boolean matrix
    basket_matrix = csr_matrix(
        (
            np.ones(len(filtered_items), dtype=bool),
            (row_indices, col_indices),
        ),
        shape=(
            len(basket_ids),
            len(frequent_product_ids),
        ),
    )

    # 7. Convert to sparse Pandas DataFrame
    basket_encoded = pd.DataFrame.sparse.from_spmatrix(
        basket_matrix,
        index=basket_ids,
        columns=frequent_product_ids,
    )

    # FP-Growth uses string product identifiers
    basket_encoded.columns = (
        basket_encoded.columns.astype(str)
    )

    # 8. Validate matrix
    assert basket_matrix.nnz == len(filtered_items)

    assert basket_encoded.shape == (
        train_items["BASKET_ID"].nunique(),
        len(frequent_product_ids),
    )

    return basket_encoded, basket_matrix


def mine_frequent_itemsets(
    basket_encoded: pd.DataFrame,
    min_support: float = 0.002,
    max_len: int = 2,
) -> pd.DataFrame:
    """Find frequent product combinations using FP-Growth."""

    if basket_encoded.empty:
        raise ValueError("Basket-product matrix is empty.")

    if not 0 < min_support <= 1:
        raise ValueError("Invalid minimum support.")

    if max_len < 1:
        raise ValueError("max_len must be positive.")

    # 1. Run FP-Growth
    from mlxtend.frequent_patterns import fpgrowth

    frequent_itemsets = fpgrowth(
        basket_encoded,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    if frequent_itemsets.empty:
        raise ValueError("No frequent itemsets found.")

    # 2. Calculate itemset size
    frequent_itemsets["itemset_size"] = (
        frequent_itemsets["itemsets"].apply(len)
    )

    # 3. Sort by support
    frequent_itemsets = (
        frequent_itemsets
        .sort_values(
            "support",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # 4. Validate results
    if not frequent_itemsets["support"].between(0, 1).all():
        raise ValueError("Invalid itemset support values.")

    if frequent_itemsets["itemset_size"].gt(max_len).any():
        raise ValueError("Itemset exceeds maximum length.")

    return frequent_itemsets


def generate_association_rules(
    frequent_itemsets: pd.DataFrame,
    valid_catalog: pd.DataFrame,
    n_baskets: int,
    min_support: float = 0.002,
    min_confidence: float = 0.10,
    min_lift: float = 1.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate product association rules from frequent itemsets."""

    # 1. Validate parameters
    if not 0 < min_support <= 1:
        raise ValueError("Invalid minimum support.")

    if not 0 < min_confidence <= 1:
        raise ValueError("Invalid minimum confidence.")

    if min_lift <= 0:
        raise ValueError("Minimum lift must be positive.")

    if n_baskets < 1:
        raise ValueError("n_baskets must be positive.")

    # 2. Validate required columns
    required_itemsets = {"support", "itemsets"}
    required_catalog = {"PRODUCT_ID", "product_label"}

    missing_itemsets = (
        required_itemsets - set(frequent_itemsets.columns)
    )

    missing_catalog = (
        required_catalog - set(valid_catalog.columns)
    )

    if missing_itemsets or missing_catalog:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing_itemsets | missing_catalog)}"
        )

    if frequent_itemsets.empty:
        raise ValueError("Frequent itemsets are empty.")

    if not valid_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate product IDs in catalog.")

    # 3. Generate association rules
    from mlxtend.frequent_patterns import association_rules

    rules = association_rules(
        frequent_itemsets[["support", "itemsets"]],
        num_itemsets=n_baskets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    # 4. Filter rules
    rules = rules.loc[
        (rules["lift"] >= min_lift)
        & (rules["support"] >= min_support)
    ].copy()

    if rules.empty:
        raise ValueError("No association rules found.")

    # 5. Create product label mapping
    product_labels = (
        valid_catalog
        .set_index("PRODUCT_ID")["product_label"]
        .to_dict()
    )

    # 6. Convert product IDs to readable labels
    def get_itemset_labels(itemset) -> str:
        return " + ".join(
            product_labels.get(
                int(product_id),
                f"Unknown Product: {product_id}",
            )
            for product_id in sorted(itemset, key=int)
        )

    rules["antecedent_name"] = (
        rules["antecedents"].apply(get_itemset_labels)
    )

    rules["consequent_name"] = (
        rules["consequents"].apply(get_itemset_labels)
    )

    # 7. Prepare readable association results
    association_results = (
        rules[
            [
                "antecedent_name",
                "consequent_name",
                "support",
                "confidence",
                "lift",
            ]
        ]
        .sort_values(
            ["lift", "confidence"],
            ascending=False,
        )
    )

    return rules, association_results


def build_recommendation_rules(
    rules: pd.DataFrame,
    valid_catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare association rules for product recommendations."""

    # 1. Validate required columns
    required_rules = {
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
    }

    required_catalog = {
        "PRODUCT_ID",
        "product_label",
    }

    missing_rules = required_rules - set(rules.columns)
    missing_catalog = required_catalog - set(valid_catalog.columns)

    if missing_rules or missing_catalog:
        raise ValueError(
            f"Missing columns: {sorted(missing_rules | missing_catalog)}"
        )

    if rules.empty:
        raise ValueError("Association rules are empty.")

    if not valid_catalog["PRODUCT_ID"].is_unique:
        raise ValueError("Duplicate product IDs in catalog.")

    # 2. Keep rules with one antecedent and one consequent
    pair_rules = rules.loc[
        rules["antecedents"].apply(len).eq(1)
        & rules["consequents"].apply(len).eq(1)
    ].copy()

    if pair_rules.empty:
        raise ValueError("No single-product association rules found.")

    # 3. Extract product IDs
    pair_rules["antecedent_id"] = (
        pair_rules["antecedents"]
        .apply(lambda items: int(next(iter(items))))
    )

    pair_rules["consequent_id"] = (
        pair_rules["consequents"]
        .apply(lambda items: int(next(iter(items))))
    )

    # 4. Prepare recommendation rules
    recommendation_rules = pair_rules[
        [
            "antecedent_id",
            "consequent_id",
            "support",
            "confidence",
            "lift",
        ]
    ].copy()

    # 5. Add recommended product labels
    recommendation_rules = recommendation_rules.merge(
        valid_catalog[
            ["PRODUCT_ID", "product_label"]
        ].rename(
            columns={
                "PRODUCT_ID": "consequent_id",
                "product_label": "recommended_product",
            }
        ),
        on="consequent_id",
        how="left",
        validate="many_to_one",
    )

    # 6. Validate recommendation rules
    if recommendation_rules["recommended_product"].isna().any():
        raise ValueError("Missing recommended product labels.")

    if (
        recommendation_rules["antecedent_id"]
        == recommendation_rules["consequent_id"]
    ).any():
        raise ValueError("A product cannot recommend itself.")

    return recommendation_rules


def recommend_products(
    product_id: int,
    recommendation_rules: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend related products using association rules."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    required = {
        "antecedent_id",
        "consequent_id",
        "recommended_product",
        "support",
        "confidence",
        "lift",
    }

    missing = required - set(recommendation_rules.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # 1. Find rules starting with the selected product
    recommendations = recommendation_rules.loc[
        recommendation_rules["antecedent_id"] == product_id
    ].copy()

    # 2. Rank by Confidence, Lift and Support
    recommendations = recommendations.sort_values(
        ["confidence", "lift", "support"],
        ascending=False,
    )

    # 3. Remove duplicate recommended products
    recommendations = recommendations.drop_duplicates(
        subset="consequent_id"
    )

    # 4. Return top recommendations
    return (
        recommendations[
            [
                "consequent_id",
                "recommended_product",
                "support",
                "confidence",
                "lift",
            ]
        ]
        .head(top_n)
        .reset_index(drop=True)
    )


def recommend_association_customer(
    household_key: int,
    evaluation_rules: pd.DataFrame,
    train_purchase_sets: Mapping[int, set[int]],
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend new products using customer purchase history and rules."""

    if top_n < 1:
        raise ValueError("top_n must be positive.")

    if household_key not in train_purchase_sets:
        raise ValueError("Customer is not available.")

    required = {
        "antecedent_id",
        "consequent_id",
        "confidence",
        "lift",
        "support",
    }

    missing = required - set(evaluation_rules.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # 1. Get previously purchased products
    purchased_ids = train_purchase_sets[household_key]

    # 2. Find matching association rules
    candidates = evaluation_rules.loc[
        evaluation_rules["antecedent_id"].isin(purchased_ids)
        & ~evaluation_rules["consequent_id"].isin(purchased_ids)
    ].copy()

    # 3. Rank by confidence, lift and support
    candidates = (
        candidates
        .sort_values(
            ["confidence", "lift", "support"],
            ascending=False,
        )
        .drop_duplicates("consequent_id")
        .head(top_n)
    )

    # 4. Prepare recommendation results
    recommendations = candidates.rename(
        columns={
            "consequent_id": "PRODUCT_ID",
            "confidence": "recommendation_score",
        }
    )[
        [
            "PRODUCT_ID",
            "recommendation_score",
            "lift",
            "support",
        ]
    ].copy()

    return recommendations.reset_index(drop=True)


