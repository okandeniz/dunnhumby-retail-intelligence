from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(filename: str) -> pd.DataFrame:
    """Load CSV files as strings."""
    return pd.read_csv(
        RAW_DIR / filename,
        dtype="string",
        encoding="utf-8-sig",
    )

def main() -> None:
    """Validate coupon duplicates and relationships."""

    coupon = load_csv("coupon.csv")
    redemption = load_csv("coupon_redempt.csv")
    campaign_table = load_csv("campaign_table.csv")

    # 1. Check exact duplicate coupon records
    key_columns = [
        "COUPON_UPC",
        "PRODUCT_ID",
        "CAMPAIGN",
    ]

    duplicate_count = coupon.duplicated(
        subset=key_columns
    ).sum()

    duplicate_groups = (
        coupon.groupby(key_columns, dropna=False)
        .size()
        .reset_index(name="frequency")
        .query("frequency > 1")
        .sort_values("frequency", ascending=False)
    )

    print("\nCOUPON DUPLICATE ANALYSIS")
    print(f"Total rows: {len(coupon):,}")
    print(f"Duplicate rows: {duplicate_count:,}")
    print(f"Duplicate groups: {len(duplicate_groups):,}")
    print(duplicate_groups.head(10).to_string(index=False))

    # 2. Check coupon redemption against
    # both coupon identifier and campaign

    valid_coupon_pairs = coupon[
        ["COUPON_UPC", "CAMPAIGN"]
    ].drop_duplicates()

    redemption_check = redemption.merge(
        valid_coupon_pairs,
        on=["COUPON_UPC", "CAMPAIGN"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )

    unmatched_coupons = (
        redemption_check["_merge"] == "left_only"
    ).sum()

    # 3. Check household campaign participation
    valid_household_campaigns = campaign_table[
        ["household_key", "CAMPAIGN"]
    ].drop_duplicates()

    participation_check = redemption.merge(
        valid_household_campaigns,
        on=["household_key", "CAMPAIGN"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )

    unmatched_participation = (
        participation_check["_merge"] == "left_only"
    ).sum()

    print("\nCOUPON RELATIONSHIP VALIDATION")
    print(f"Unmatched coupon redemptions: {unmatched_coupons:,}")
    print(f"Unmatched household campaigns: {unmatched_participation:,}")

    # 4. Export duplicate report
    duplicate_groups.to_csv(
        REPORT_DIR / "coupon_duplicate_profile.csv",
        index=False,
    )

    print("\nReport saved successfully.")


if __name__ == "__main__":
    main()