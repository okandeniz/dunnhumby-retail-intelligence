from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 200_000

def load_csv(filename: str) -> pd.DataFrame:
    """Load a small CSV file for key validation."""
    return pd.read_csv(
        RAW_DIR / filename,
        dtype="string",
        encoding="utf-8-sig",
    )

def check_key(
    df: pd.DataFrame,
    table_name: str,
    columns: list[str],
) -> dict:
    """Check null and duplicate candidate keys."""

    return {
        "table": table_name,
        "candidate_key": ", ".join(columns),
        "rows": len(df),
        "null_key_rows": int(
            df[columns].isna().any(axis=1).sum()
        ),
        "duplicate_key_rows": int(
            df.duplicated(subset=columns).sum()
        ),
    }

def check_reference(
    name: str,
    values: pd.Series,
    valid_keys: set,
) -> dict:
    """Check values against a parent key set."""

    non_null = values.dropna()

    return {
        "relationship": name,
        "checked_rows": len(non_null),
        "unmatched_rows": int(
            (~non_null.isin(valid_keys)).sum()
        ),
    }

def main() -> None:
    """Profile candidate keys and table relationships."""

    tables = {
        "product": load_csv("product.csv"),
        "campaign_desc": load_csv("campaign_desc.csv"),
        "campaign_table": load_csv("campaign_table.csv"),
        "coupon": load_csv("coupon.csv"),
        "coupon_redempt": load_csv("coupon_redempt.csv"),
        "hh_demographic": load_csv("hh_demographic.csv"),
    }

    key_definitions = {
        "product": ["PRODUCT_ID"],
        "campaign_desc": ["CAMPAIGN"],
        "campaign_table": ["household_key", "CAMPAIGN"],
        "coupon": ["COUPON_UPC", "PRODUCT_ID", "CAMPAIGN"],
        "hh_demographic": ["household_key"],
        "coupon_redempt": [
            "household_key", "DAY", "COUPON_UPC", "CAMPAIGN"
        ],
    }

    key_results = [
        check_key(tables[name], name, columns)
        for name, columns in key_definitions.items()
    ]

    product_keys = set(
        tables["product"]["PRODUCT_ID"].dropna()
    )
    campaign_keys = set(
        tables["campaign_desc"]["CAMPAIGN"].dropna()
    )
    coupon_keys = set(
        tables["coupon"]["COUPON_UPC"].dropna()
    )

    relation_results = []

    checks = [
        (
            "campaign_table -> campaign_desc",
            tables["campaign_table"]["CAMPAIGN"],
            campaign_keys,
        ),
        (
            "coupon -> campaign_desc",
            tables["coupon"]["CAMPAIGN"],
            campaign_keys,
        ),
        (
            "coupon -> product",
            tables["coupon"]["PRODUCT_ID"],
            product_keys,
        ),
        (
            "coupon_redempt -> campaign_desc",
            tables["coupon_redempt"]["CAMPAIGN"],
            campaign_keys,
        ),
        (
            "coupon_redempt -> coupon",
            tables["coupon_redempt"]["COUPON_UPC"],
            coupon_keys,
        ),
    ]

    for name, values, valid_keys in checks:
        relation_results.append(
            check_reference(name, values, valid_keys)
        )

    # Read transaction keys in chunks to save memory.
    households = set()
    total_transaction_rows = 0

    reader = pd.read_csv(
        RAW_DIR / "transaction_data.csv",
        usecols=["household_key", "PRODUCT_ID"],
        dtype="string",
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        total_transaction_rows += len(chunk)

        households.update(
            chunk["household_key"].dropna().unique()
        )

        relation_results.append(
            check_reference(
                "transaction_data -> product",
                chunk["PRODUCT_ID"],
                product_keys,
            )
        )

    # Aggregate the chunk-based transaction checks.
    transaction_checks = [
        item for item in relation_results
        if item["relationship"] == "transaction_data -> product"
    ]

    relation_results = [
        item for item in relation_results
        if item["relationship"] != "transaction_data -> product"
    ]

    relation_results.append({
        "relationship": "transaction_data -> product",
        "checked_rows": sum(
            item["checked_rows"] for item in transaction_checks
        ),
        "unmatched_rows": sum(
            item["unmatched_rows"] for item in transaction_checks
        ),
    })

    demographics = set(
        tables["hh_demographic"]["household_key"].dropna()
    )

    key_report = pd.DataFrame(key_results)
    relation_report = pd.DataFrame(relation_results)

    key_report.to_csv(
        REPORT_DIR / "key_profile.csv", index=False
    )
    relation_report.to_csv(
        REPORT_DIR / "relationship_profile.csv", index=False
    )

    print("\nKEY PROFILE")
    print(key_report.to_string(index=False))

    print("\nRELATIONSHIP PROFILE")
    print(relation_report.to_string(index=False))

    print("\nHOUSEHOLD COVERAGE")
    print(f"Transaction rows: {total_transaction_rows:,}")
    print(f"Unique households: {len(households):,}")
    print(
        "Households with demographics:",
        f"{len(households & demographics):,}",
    )

    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()