"""Serve only the modes supported by the saved recommendation artifacts."""

from functools import lru_cache

import pandas as pd
from fastapi import HTTPException

from api import store
from src.recommendations.association_rules import (
    recommend_association_customer,
    recommend_products,
)
from src.recommendations.content_based import (
    create_content_neighbor_lookup,
    recommend_content_based,
    recommend_content_customer,
)
from src.recommendations.item_based import (
    create_item_neighbor_lookup,
    recommend_item_based_customer,
    recommend_similar_items,
)
from src.recommendations.matrix_factorization import recommend_nmf, recommend_svd
from src.recommendations.user_based import recommend_user_based

ROOT = store.MODELS / "recommendation"
USER = ROOT / "user_based"
RULES = ROOT / "association_rules"
CUSTOMER_SCORES = {
    "Item-Based CF": "Sum of purchase similarities",
    "Content-Based Filtering": "Sum of product content similarities",
    "Popularity Baseline": "Customers who bought the product",
    "User-Based CF": "Share of similar customers, weighted by similarity",
    "Truncated SVD": "SVD factor score",
    "NMF": "NMF factor score",
    "Association Rules": "Rule confidence",
}
PRODUCT_SCORES = {
    "Item-Based CF": "Cosine similarity",
    "Content-Based Filtering": "Cosine similarity",
    "Association Rules": "Rule confidence",
}


def known(value: int, ids, label: str) -> None:
    if value not in ids:
        raise HTTPException(404, f"{label} was not found in this model.")


def supported(model: str, choices: dict) -> None:
    if model not in choices:
        raise HTTPException(
            422, "This model does not support this type of recommendation."
        )


def rules():
    return store.artifact(RULES / "recommendation_rules.joblib")


@lru_cache(maxsize=1)
def popularity() -> pd.Series:
    meta = store.artifact(USER / "user_based_metadata.joblib")
    matrix = store.matrix(USER / "customer_item_matrix.npz")
    return pd.Series(matrix.getnnz(axis=0), index=meta["product_ids"]).sort_values(
        ascending=False
    )


@lru_cache(maxsize=1)
def seeds() -> dict[int, list[int]]:
    data = store.metadata(store.DATA / "recommendation/customer_seed_products.json")
    return {int(customer): items for customer, items in data.items()}


@lru_cache(maxsize=2)
def neighbors(model: str):
    if model == "Item-Based CF":
        folder = ROOT / "item_based"
        ids = pd.Index(
            store.artifact(folder / "item_based_metadata.joblib")["product_ids"]
        )
        return create_item_neighbor_lookup(
            store.artifact(folder / "item_knn.joblib"),
            store.matrix(folder / "item_user_matrix.npz"),
            ids,
            30,
        )
    folder = ROOT / "content_based"
    ids = pd.Index(
        store.artifact(folder / "content_based_metadata.joblib")["product_ids"]
    )
    return create_content_neighbor_lookup(
        store.artifact(folder / "content_knn.joblib"),
        store.matrix(folder / "content_matrix.npz"),
        ids,
        30,
    )


def customer_ranking(model: str, customer_id: int, top_n: int) -> pd.DataFrame:
    """Rank unseen products with the selected method and its training history."""
    supported(model, CUSTOMER_SCORES)
    if model == "Association Rules":
        purchases = store.artifact(RULES / "train_purchase_sets.joblib")
        known(customer_id, purchases, "Customer")
        result = recommend_association_customer(customer_id, rules(), purchases, top_n)
        labels = rules()[["consequent_id", "recommended_product"]].drop_duplicates(
            "consequent_id"
        )
        return result.merge(
            labels, left_on="PRODUCT_ID", right_on="consequent_id"
        ).rename(
            columns={
                "recommended_product": "product_label",
                "recommendation_score": "score",
            }
        )

    if model in ("Truncated SVD", "NMF"):
        name = "svd" if model == "Truncated SVD" else "nmf"
        folder = ROOT / name
        data = store.factors(folder / f"{name}_factors.npz")
        customers = pd.Index(data["customer_ids"])
        known(customer_id, customers, "Customer")
        recommend = recommend_svd if name == "svd" else recommend_nmf
        result = recommend(
            customer_id,
            data["user_factors"],
            data["item_factors"],
            store.matrix(folder / "customer_item_matrix.npz"),
            customers,
            pd.Index(data["product_ids"]),
            store.csv(folder / "product_catalog.csv"),
            top_n,
        )
    else:
        meta = store.artifact(USER / "user_based_metadata.joblib")
        customers = pd.Index(meta["customer_ids"])
        known(customer_id, customers, "Customer")
        products = pd.Index(meta["product_ids"])
        matrix = store.matrix(USER / "customer_item_matrix.npz")
        catalog = store.csv(USER / "product_catalog.csv")
        if model in ("Item-Based CF", "Content-Based Filtering"):
            recommend = (
                recommend_item_based_customer
                if model == "Item-Based CF"
                else recommend_content_customer
            )
            result = recommend(
                customer_id,
                matrix,
                customers,
                products,
                seeds(),
                neighbors(model),
                top_n,
            )
            return result.merge(catalog, on="PRODUCT_ID").rename(
                columns={"recommendation_score": "score"}
            )
        if model == "Popularity Baseline":
            seen = products.take(matrix[customers.get_loc(customer_id)].indices)
            ranking = popularity().drop(seen).head(top_n)
            result = ranking.rename_axis("PRODUCT_ID").reset_index(name="score")
            return result.merge(catalog, on="PRODUCT_ID", validate="one_to_one")
        result = recommend_user_based(
            customer_id,
            store.artifact(USER / "user_knn.joblib"),
            matrix,
            customers,
            products,
            catalog,
            top_n,
            n_neighbors=30,
        )
    return result.rename(columns={"recommendation_score": "score"})


def catalog(model: str) -> pd.DataFrame:
    supported(model, PRODUCT_SCORES)
    if model == "Content-Based Filtering":
        return store.csv(ROOT / "content_based/content_product_catalog.csv")
    frame = store.csv(ROOT / "item_based/product_catalog.csv")
    if model == "Association Rules":
        frame = frame.loc[frame.PRODUCT_ID.isin(rules().antecedent_id)]
    return frame


def product_ranking(model: str, product_id: int, top_n: int) -> pd.DataFrame:
    """Find related products using the saved neighbors or association rules."""
    frame = catalog(model)
    known(product_id, frame.PRODUCT_ID.values, "Product")
    if model == "Association Rules":
        return recommend_products(product_id, rules(), top_n).rename(
            columns={
                "consequent_id": "PRODUCT_ID",
                "recommended_product": "product_label",
                "confidence": "score",
            }
        )
    if model == "Item-Based CF":
        folder = ROOT / "item_based"
        products = pd.Index(
            store.artifact(folder / "item_based_metadata.joblib")["product_ids"]
        )
        result = recommend_similar_items(
            product_id,
            store.artifact(folder / "item_knn.joblib"),
            store.matrix(folder / "item_user_matrix.npz"),
            products,
            frame,
            top_n,
        )
    else:
        folder = ROOT / "content_based"
        products = pd.Index(
            store.artifact(folder / "content_based_metadata.joblib")["product_ids"]
        )
        result = recommend_content_based(
            product_id,
            store.artifact(folder / "content_knn.joblib"),
            store.matrix(folder / "content_matrix.npz"),
            products,
            frame,
            top_n,
        )
    return result.rename(columns={"cosine_similarity": "score"})
