from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

CHUNK_SIZE = 100_000
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TABLES = [
    "campaign_desc",
    "campaign_table",
    "causal_data",
    "coupon",
    "coupon_redempt",
    "hh_demographic",
    "product",
    "transaction_data",
]

def get_mysql_schema(table_name: str) -> dict:
    """Read column definitions from MySQL."""

    query = text("""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            COLUMN_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION,
            NUMERIC_SCALE,
            IS_NULLABLE,
            EXTRA
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = :table_name
    """)

    with get_engine().connect() as conn:
        rows = conn.execute(
            query, {"table_name": table_name}
        ).mappings().all()

    return {
        row["COLUMN_NAME"]: dict(row)
        for row in rows
        if "auto_increment" not in row["EXTRA"].lower()
    }

def validate_column(
    values: pd.Series,
    metadata: dict,
) -> dict:
    """Validate values against the MySQL column type."""

    data_type = metadata["DATA_TYPE"]
    non_null = values.dropna().str.strip()

    result = {
        "null_count": int(values.isna().sum()),
        "null_violations": 0,
        "invalid_type": 0,
        "out_of_range": 0,
        "length_exceeded": 0,
        "rounding_required": 0,
    }

    if metadata["IS_NULLABLE"] == "NO":
        result["null_violations"] = result["null_count"]

    if data_type in ("int", "bigint"):
        valid = non_null.str.fullmatch(
            r"[+-]?\d+"
        ).fillna(False)

        numbers = pd.to_numeric(
            non_null.where(valid), errors="coerce"
        )

        result["invalid_type"] = int(
            (~valid | numbers.isna()).sum()
        )

        if data_type == "int":
            minimum, maximum = -(2**31), 2**31 - 1
        else:
            minimum, maximum = -(2**63), 2**63 - 1

        if "unsigned" in metadata["COLUMN_TYPE"].lower():
            minimum = 0
            maximum = 2**(32 if data_type == "int" else 64) - 1

        result["out_of_range"] = int(
            ((numbers < minimum) | (numbers > maximum)).sum()
        )

    elif data_type == "decimal":
        precision = int(metadata["NUMERIC_PRECISION"])
        scale = int(metadata["NUMERIC_SCALE"])

        quantum = Decimal(1).scaleb(-scale)
        max_absolute = Decimal(10) ** (precision - scale)

        # Process unique values to reduce repeated conversions.
        value_counts = non_null.value_counts()

        for raw_value, frequency in value_counts.items():
            frequency = int(frequency)

            try:
                number = Decimal(raw_value)
            except InvalidOperation:
                result["invalid_type"] += frequency
                continue

            if not number.is_finite():
                result["invalid_type"] += frequency
                continue

            if abs(number) >= max_absolute:
                result["out_of_range"] += frequency
                continue

            rounded = number.quantize(
                quantum,
                rounding=ROUND_HALF_UP,
            )

            if abs(rounded) >= max_absolute:
                result["out_of_range"] += frequency
                continue

            if number != rounded:
                result["rounding_required"] += frequency

    elif data_type in ("varchar", "char"):
        max_length = int(
            metadata["CHARACTER_MAXIMUM_LENGTH"]
        )

        result["length_exceeded"] = int(
            (non_null.str.len() > max_length).sum()
        )

    return result


def profile_table(name: str) -> list[dict]:
    """Validate a CSV file against its MySQL schema."""

    file_path = RAW_DIR / f"{name}.csv"
    table_name = f"raw_{name}"

    schema = get_mysql_schema(table_name)

    if not schema:
        raise ValueError(
            f"Missing MySQL table: {table_name}"
        )

    csv_columns = pd.read_csv(
        file_path, nrows=0
    ).columns.tolist()

    if set(csv_columns) != set(schema):
        raise ValueError(
            f"Column mismatch in {table_name}\n"
            f"CSV columns: {csv_columns}\n"
            f"MySQL columns: {list(schema)}"
        )

    # Validate only a sample of the largest file.
    sample_only = name == "causal_data"
    row_limit = 100_000 if sample_only else None

    results = {
        column: {
            "table": table_name,
            "column": column,
            "mysql_type": schema[column]["COLUMN_TYPE"],
            "checked_rows": 0,
            "null_count": 0,
            "null_violations": 0,
            "invalid_type": 0,
            "out_of_range": 0,
            "length_exceeded": 0,
            "rounding_required": 0,
            "scope": "sample" if sample_only else "full",
        }
        for column in csv_columns
    }

    reader = pd.read_csv(
        file_path,
        dtype="string",
        chunksize=CHUNK_SIZE,
        nrows=row_limit,
        encoding="utf-8-sig",
    )

    for chunk in reader:
        for column in csv_columns:
            validation = validate_column(
                chunk[column], schema[column]
            )

            results[column]["checked_rows"] += len(chunk)

            for metric, value in validation.items():
                results[column][metric] += value

    checked_rows = results[csv_columns[0]]["checked_rows"]

    print(
        f"{table_name}: {checked_rows:,} rows checked"
        + (" (sample)" if sample_only else "")
    )

    return list(results.values())


def main() -> None:
    """Generate the data type validation report."""

    all_results = []

    for name in TABLES:
        all_results.extend(profile_table(name))

    report = pd.DataFrame(all_results)

    error_columns = [
        "null_violations",
        "invalid_type",
        "out_of_range",
        "length_exceeded",
    ]

    report["status"] = report[error_columns].sum(
        axis=1
    ).eq(0).map({True: "PASS", False: "FAIL"})

    report.to_csv(
        REPORT_DIR / "dtype_validation.csv",
        index=False,
    )

    print("\nDATA TYPE VALIDATION SUMMARY")
    print(report["status"].value_counts().to_string())

    failed = report[report["status"] == "FAIL"]

    if not failed.empty:
        print("\nFAILED COLUMNS")
        print(failed.to_string(index=False))
    else:
        print("\nAll checked columns passed validation.")

    print("\nReport saved to reports/dtype_validation.csv")


if __name__ == "__main__":
    main()