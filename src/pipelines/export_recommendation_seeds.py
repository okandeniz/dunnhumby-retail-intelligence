"""Save recent training products for the two customer-level neighbor methods."""

import json
from pathlib import Path

import pandas as pd

from src.recommendations.item_based import build_customer_seed_products

ROOT = Path(__file__).resolve().parents[2]


def export() -> Path:
    folder = ROOT / "data/processed/recommendation"
    metadata = json.loads((folder / "evaluation_metadata.json").read_text())
    catalog = pd.read_csv(ROOT / "models/recommendation/user_based/product_catalog.csv")
    items = pd.read_csv(
        ROOT / "data/raw/transaction_data.csv",
        usecols=["household_key", "PRODUCT_ID", "DAY", "QUANTITY", "SALES_VALUE"],
    )
    eligible = (
        items.DAY.between(1, metadata["train_end_day"])
        & items.QUANTITY.gt(0)
        & items.SALES_VALUE.gt(0)
    )
    history = items.loc[eligible].rename(columns={"DAY": "basket_day"})
    seeds = build_customer_seed_products(
        history, pd.Index(catalog.PRODUCT_ID), n_seed_items=10
    )
    path = folder / "customer_seed_products.json"
    path.write_text(json.dumps(seeds), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(export())
