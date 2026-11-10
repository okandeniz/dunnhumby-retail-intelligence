"""Prepare store sales shares from the history available at the forecast origin."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def export(root: Path = ROOT) -> Path:
    folder = root / "data/processed/forecasting"
    products = pd.read_csv(
        folder / "future_forecast.csv", usecols=["PRODUCT_ID", "DAY"]
    )
    keys = ["PRODUCT_ID", "STORE_ID", "DAY"]
    totals = []
    for chunk in pd.read_csv(
        root / "data/raw/transaction_data.csv",
        usecols=[*keys, "QUANTITY"],
        chunksize=250_000,
    ):
        rows = chunk.rename(columns={"DAY": "sale_day"}).merge(
            products, on="PRODUCT_ID", validate="many_to_one"
        )
        recent = rows.loc[
            (rows.DAY - rows.sale_day).between(0, 179) & rows.QUANTITY.gt(0)
        ]
        if recent.STORE_ID.isna().any():
            raise ValueError("A recent sale is missing its store ID.")
        totals.append(recent.groupby(keys).QUANTITY.sum())

    sales = pd.concat(totals).groupby(level=keys).sum().rename("units_last_180d")
    path = folder / "store_sales_180d.csv"
    with NamedTemporaryFile(mode="w", dir=folder, delete=False, newline="") as tmp:
        sales.to_csv(tmp)
    Path(tmp.name).replace(path)
    return path


if __name__ == "__main__":
    print(export())
