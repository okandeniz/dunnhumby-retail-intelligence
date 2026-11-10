
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = PROJECT_ROOT / "data/raw/transaction_data.csv"

TABLE_NAME = "raw_transaction_data"
CHUNK_SIZE = 50_000
INSERT_BATCH_SIZE = 2_000
EXPECTED_ROWS = 2_595_732

MONEY_COLUMNS = [
    "SALES_VALUE",
    "RETAIL_DISC",
    "COUPON_DISC",
    "COUPON_MATCH_DISC",
]

QUANTUM = Decimal("0.0001")


def convert_decimal(value):
    """Convert a numeric value to DECIMAL(18,4)."""

    if pd.isna(value):
        return None

    number = Decimal(str(value))

    if not number.is_finite():
        raise ValueError(f"Invalid decimal value: {value}")

    return number.quantize(
        QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def get_load_status(engine):
    """Get the current loading status."""

    query = text(f"""
        SELECT
            COUNT(*) AS total_rows,
            MIN(row_id) AS min_id,
            MAX(row_id) AS max_id
        FROM {TABLE_NAME}
    """)

    with engine.connect() as conn:
        return conn.execute(query).mappings().one()


def main():
    """Load transactions in resumable batches."""

    engine = get_engine()

    status = get_load_status(engine)
    existing_rows = int(status["total_rows"])

    print(f"Existing rows: {existing_rows:,}")

    # Validate existing row IDs before resuming.
    if existing_rows:
        if (
            status["min_id"] != 1
            or status["max_id"] != existing_rows
        ):
            raise ValueError(
                "Existing row IDs are not continuous."
            )

    if existing_rows == EXPECTED_ROWS:
        print("SKIPPED: Table already fully loaded.")
        return

    if existing_rows > EXPECTED_ROWS:
        raise ValueError("Too many existing rows.")

    if existing_rows % CHUNK_SIZE != 0:
        raise ValueError(
            "Existing rows do not match a completed batch."
        )

    reader = pd.read_csv(
        FILE_PATH,
        dtype="string",
        chunksize=CHUNK_SIZE,
        encoding="utf-8-sig",
    )

    current_row = 0

    for chunk in reader:
        start_row = current_row
        current_row += len(chunk)

        # Skip previously committed batches.
        if current_row <= existing_rows:
            continue

        if start_row < existing_rows:
            raise ValueError(
                "Cannot resume from a partial batch."
            )

        # Preserve source row position as row_id.
        chunk.insert(
            0,
            "row_id",
            range(start_row + 1, current_row + 1),
        )

        # Prepare decimal fields.
        for column in MONEY_COLUMNS:
            chunk[column] = chunk[column].map(
                convert_decimal
            )

        # Convert missing values to SQL NULL.
        chunk = chunk.astype(object).where(
            chunk.notna(), None
        )

        # Commit each completed batch.
        with engine.begin() as conn:
            chunk.to_sql(
                name=TABLE_NAME,
                con=conn,
                if_exists="append",
                index=False,
                chunksize=INSERT_BATCH_SIZE,
            )

        print(
            f"Loaded: {current_row:,} / "
            f"{EXPECTED_ROWS:,}"
        )

    # Final validation
    final_status = get_load_status(engine)
    mysql_rows = int(final_status["total_rows"])

    if mysql_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Row count mismatch: "
            f"MySQL={mysql_rows:,}, "
            f"CSV={EXPECTED_ROWS:,}"
        )

    if final_status["max_id"] != EXPECTED_ROWS:
        raise ValueError("Final row ID mismatch.")

    print("\nTRANSACTION DATA LOADING: SUCCESS")
    print(f"Total rows: {mysql_rows:,}")


if __name__ == "__main__":
    main()
