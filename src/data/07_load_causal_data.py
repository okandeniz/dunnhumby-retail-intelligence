
import csv
import hashlib
import json
import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = PROJECT_ROOT / "data/raw/causal_data.csv"
TEMP_DIR = PROJECT_ROOT / "data/processed"
REPORT_DIR = PROJECT_ROOT / "reports"

TEMP_FILE = TEMP_DIR / "_causal_chunk.csv"
MANIFEST_FILE = REPORT_DIR / "causal_load_manifest.json"

TABLE_NAME = "raw_causal_data"
EXPECTED_ROWS = 36_786_524
CHUNK_SIZE = 250_000

COLUMNS = [
    "PRODUCT_ID",
    "STORE_ID",
    "WEEK_NO",
    "display",
    "mailer",
]

INT_MIN = -(2**31)
INT_MAX = 2**31 - 1

load_dotenv(PROJECT_ROOT / ".env")

TEMP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


LOAD_SQL = r"""
LOAD DATA LOCAL INFILE %s
INTO TABLE raw_causal_data
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
ESCAPED BY ''
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(
    @product_id,
    @store_id,
    @week_no,
    @display_code,
    @mailer_code
)
SET
    PRODUCT_ID = NULLIF(@product_id, ''),
    STORE_ID = NULLIF(@store_id, ''),
    WEEK_NO = NULLIF(@week_no, ''),
    `display` = NULLIF(@display_code, ''),
    `mailer` = NULLIF(@mailer_code, '');
"""


def get_connection():
    """Create a MySQL connection for bulk loading."""

    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
        charset="utf8mb4",
        local_infile=True,
        autocommit=False,
        read_timeout=600,
        write_timeout=600,
    )


def get_row_count(connection):
    """Read the current number of imported rows."""

    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT COUNT(*) FROM {TABLE_NAME}"
        )
        return int(cursor.fetchone()[0])


def get_file_signature():
    """Identify the source file for safe resuming."""

    with FILE_PATH.open("rb") as source:
        file_hash = hashlib.file_digest(
            source, "sha256"
        ).hexdigest()

    return {
        "sha256": file_hash,
        "chunk_size": CHUNK_SIZE,
        "expected_rows": EXPECTED_ROWS,
    }


def verify_manifest(existing_rows):
    """Prevent resuming with a different source file."""

    signature = get_file_signature()

    if MANIFEST_FILE.exists():
        previous = json.loads(
            MANIFEST_FILE.read_text(encoding="utf-8")
        )

        if previous != signature:
            raise ValueError(
                "Source file or chunk size has changed."
            )

    elif existing_rows > 0:
        raise ValueError(
            "Existing rows found without a load manifest."
        )

    else:
        MANIFEST_FILE.write_text(
            json.dumps(signature, indent=2),
            encoding="utf-8",
        )


def validate_row(row, row_number):
    """Validate source fields before MySQL insertion."""

    if len(row) != len(COLUMNS):
        raise ValueError(
            f"Invalid column count at row {row_number}"
        )

    for value in row[:3]:
        if value == "":
            continue

        try:
            number = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid integer at row {row_number}"
            ) from exc

        if not INT_MIN <= number <= INT_MAX:
            raise ValueError(
                f"Integer out of range at row {row_number}"
            )

    for value in row[3:]:
        if len(value) > 10:
            raise ValueError(
                f"Text too long at row {row_number}"
            )


def create_chunk(reader, start_row):
    """Write the next validated CSV chunk to disk."""

    row_count = 0

    with TEMP_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(
            output, lineterminator="\n"
        )
        writer.writerow(COLUMNS)

        for _ in range(CHUNK_SIZE):
            row = next(reader, None)

            if row is None:
                break

            validate_row(
                row, start_row + row_count + 1
            )

            writer.writerow(row)
            row_count += 1

    return row_count


def load_chunk(connection, expected_count):
    """Load and commit one CSV chunk."""

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                LOAD_SQL,
                (TEMP_FILE.resolve().as_posix(),),
            )

            inserted = cursor.rowcount

            # LOCAL INFILE can turn errors into warnings.
            cursor.execute("SHOW COUNT(*) WARNINGS")
            warning_count = int(cursor.fetchone()[0])

            if warning_count:
                cursor.execute("SHOW WARNINGS LIMIT 5")
                warnings = cursor.fetchall()

                raise ValueError(
                    f"MySQL warnings: {warning_count}\n"
                    f"Examples: {warnings}"
                )

            if inserted != expected_count:
                raise ValueError(
                    f"Expected {expected_count} rows, "
                    f"inserted {inserted}."
                )

        connection.commit()

    except Exception:
        connection.rollback()
        raise


def main():
    """Load causal data using resumable batches."""

    if not FILE_PATH.exists():
        raise FileNotFoundError(FILE_PATH)

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT @@GLOBAL.local_infile")
            local_infile = cursor.fetchone()[0]

        if str(local_infile).upper() not in ("1", "ON"):
            raise RuntimeError(
                "MySQL local_infile is disabled."
            )

        existing_rows = get_row_count(connection)

        print(f"Existing rows: {existing_rows:,}")

        if existing_rows > EXPECTED_ROWS:
            raise ValueError("Too many existing rows.")

        if (
            existing_rows != EXPECTED_ROWS
            and existing_rows % CHUNK_SIZE != 0
        ):
            raise ValueError(
                "Existing rows do not match "
                "a completed batch."
            )

        verify_manifest(existing_rows)

        if existing_rows == EXPECTED_ROWS:
            print("SKIPPED: Already fully loaded.")
            return

        with FILE_PATH.open(
            "r", newline="", encoding="utf-8-sig"
        ) as source:

            reader = csv.reader(source, strict=True)

            header = next(reader)

            if header != COLUMNS:
                raise ValueError(
                    f"Unexpected CSV columns: {header}"
                )

            # Skip previously committed source records.
            for _ in range(existing_rows):
                if next(reader, None) is None:
                    raise ValueError(
                        "CSV ended before existing row count."
                    )

            total_loaded = existing_rows

            while True:
                chunk_rows = create_chunk(
                    reader, total_loaded
                )

                if chunk_rows == 0:
                    break

                load_chunk(connection, chunk_rows)

                total_loaded += chunk_rows

                print(
                    f"Loaded: {total_loaded:,} / "
                    f"{EXPECTED_ROWS:,}",
                    flush=True,
                )

                TEMP_FILE.unlink(missing_ok=True)

        final_rows = get_row_count(connection)

        if final_rows != EXPECTED_ROWS:
            raise ValueError(
                f"Row count mismatch: {final_rows:,}"
            )

        print("\nCAUSAL DATA LOADING: SUCCESS")
        print(f"Total rows: {final_rows:,}")

    finally:
        TEMP_FILE.unlink(missing_ok=True)
        connection.close()


if __name__ == "__main__":
    main()
