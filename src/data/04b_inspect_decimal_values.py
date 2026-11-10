
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "transaction_data.csv"
)

COLUMNS = ["SALES_VALUE", "RETAIL_DISC"]
CHUNK_SIZE = 100_000

PATTERN = r"[+-]?\d+(?:\.\d+)?"


def inspect_invalid_values() -> None:
    """Inspect values rejected by decimal validation."""

    invalid_counts = {
        column: Counter()
        for column in COLUMNS
    }

    reader = pd.read_csv(
        FILE_PATH,
        usecols=COLUMNS,
        dtype="string",
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
    )

    for chunk in reader:
        for column in COLUMNS:
            values = chunk[column].dropna().str.strip()

            valid = values.str.fullmatch(PATTERN).fillna(False)
            invalid = values[~valid]

            invalid_counts[column].update(
                invalid.value_counts().to_dict()
            )

    for column, counter in invalid_counts.items():
        print(f"\nCOLUMN: {column}")
        print(f"Invalid rows: {sum(counter.values()):,}")
        print(f"Unique invalid values: {len(counter):,}")

        print("\nINVALID VALUE EXAMPLES")

        for value, frequency in counter.most_common(15):
            try:
                numeric_value = Decimal(value)
                is_numeric = numeric_value.is_finite()
            except InvalidOperation:
                is_numeric = False

            print(
                f"Value: {value} | "
                f"Count: {frequency} | "
                f"Numeric: {is_numeric}"
            )


if __name__ == "__main__":
    inspect_invalid_values()
