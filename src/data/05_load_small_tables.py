from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

CHUNK_SIZE = 20_000
INSERT_BATCH_SIZE = 1_000

TABLES = [
    "campaign_desc",
    "campaign_table",
    "coupon",
    "coupon_redempt",
    "hh_demographic",
    "product",
]

def get_expected_counts() -> dict:
    """Read expected row counts from the inventory report."""

    inventory = pd.read_csv(
        REPORT_DIR / "data_inventory.csv"
    )

    return dict(
        zip(inventory["file_name"], inventory["rows"])
    )

def get_table_count(connection, table_name: str) -> int:
    """Return the number of rows in a MySQL table."""

    query = text(f"SELECT COUNT(*) FROM `{table_name}`")
    return connection.execute(query).scalar_one()

def load_table(name: str, expected_rows: int) -> None:
    """Load a CSV into MySQL using one transaction."""

    engine = get_engine()

    table_name = f"raw_{name}"
    file_path = RAW_DIR / f"{name}.csv"

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    print(f"\nLoading: {table_name}")

    with engine.begin() as connection:
        existing_rows = get_table_count(
            connection, table_name
        )

        if existing_rows == expected_rows:
            print("  SKIPPED: Table already loaded.")
            return

        if existing_rows > 0:
            raise ValueError(
                f"{table_name} contains {existing_rows:,} rows. "
                "Resolve existing data before reloading."
            )

        reader = pd.read_csv(
            file_path,
            dtype="string",
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
        )

        loaded_rows = 0

        for chunk in reader:
            # Convert pandas missing values to SQL NULL.
            chunk = chunk.astype(object).where(
                chunk.notna(), None
            )

            chunk.to_sql(
                name=table_name,
                con=connection,
                if_exists="append",
                index=False,
                chunksize=INSERT_BATCH_SIZE,
            )

            loaded_rows += len(chunk)

            print(
                f"  Loaded: {loaded_rows:,} / "
                f"{expected_rows:,}"
            )

        mysql_rows = get_table_count(
            connection, table_name
        )

        if mysql_rows != expected_rows:
            raise ValueError(
                f"Row count mismatch in {table_name}: "
                f"expected={expected_rows}, "
                f"actual={mysql_rows}"
            )

    print(f"  SUCCESS: {mysql_rows:,} rows")

def main() -> None:
    """Load and validate small raw tables."""

    expected_counts = get_expected_counts()

    for name in TABLES:
        filename = f"{name}.csv"

        load_table(
            name=name,
            expected_rows=int(expected_counts[filename]),
        )

    print("\nSMALL TABLE LOADING COMPLETED")


if __name__ == "__main__":
    main()