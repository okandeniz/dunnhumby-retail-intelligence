from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000
SAMPLE_SIZE = 10_000

def profile_csv(file_path: Path):
    """Profile a CSV file without loading it fully into memory."""

    print(f"\nProcessing: {file_path.name}")

    # Sample for initial data type inference
    sample = pd.read_csv(
        file_path,
        nrows=SAMPLE_SIZE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    row_count = 0
    missing_counts = pd.Series(
        0, index=sample.columns, dtype="int64"
    )

    # Process the full file in chunks
    reader = pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        dtype="string",
        encoding="utf-8-sig",
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        row_count += len(chunk)
        missing_counts = missing_counts.add(
            chunk.isna().sum(), fill_value=0
        )

        if chunk_number % 25 == 0:
            print(f"  Rows processed: {row_count:,}")

    file_summary = {
        "file_name": file_path.name,
        "rows": row_count,
        "columns": len(sample.columns),
        "size_mb": round(
            file_path.stat().st_size / (1024**2), 2
        ),
    }

    column_profiles = []

    for column in sample.columns:
        missing_count = int(missing_counts[column])

        column_profiles.append(
            {
                "file_name": file_path.name,
                "column": column,
                "sample_dtype": str(sample[column].dtype),
                "missing_count": missing_count,
                "missing_pct": round(
                    missing_count / row_count * 100, 2
                ) if row_count else 0,
                "sample_unique": sample[column].nunique(),
            }
        )

    return file_summary, column_profiles

def main():
    """Generate data inventory reports."""

    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    file_summaries = []
    all_column_profiles = []

    for file_path in files:
        summary, columns = profile_csv(file_path)

        file_summaries.append(summary)
        all_column_profiles.extend(columns)

    # Save reports
    inventory = pd.DataFrame(file_summaries)
    column_profile = pd.DataFrame(all_column_profiles)

    inventory.to_csv(
        REPORT_DIR / "data_inventory.csv",
        index=False,
    )

    column_profile.to_csv(
        REPORT_DIR / "column_profile.csv",
        index=False,
    )

    print("\nDATA INVENTORY")
    print(inventory.to_string(index=False))

    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()