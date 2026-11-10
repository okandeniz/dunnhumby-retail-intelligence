
import pandas as pd


def build_campaign_participation(
    campaign_table: pd.DataFrame,
    campaign_desc: pd.DataFrame,
    coupon_redempt: pd.DataFrame,
) -> pd.DataFrame:
    """Build household-campaign redemption records."""

    # 1. Validate required columns
    required = [
        (campaign_table, {"CAMPAIGN", "household_key"}),
        (campaign_desc, {"CAMPAIGN", "DESCRIPTION"}),
        (coupon_redempt, {"CAMPAIGN", "household_key"}),
    ]

    for data, columns in required:
        missing = columns - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

    if not campaign_desc["CAMPAIGN"].is_unique:
        raise ValueError("Duplicate campaign definitions.")

    # 2. Select unique campaign assignments
    campaign_targets = (
        campaign_table[
            ["CAMPAIGN", "household_key"]
        ]
        .drop_duplicates()
    )

    # 3. Identify coupon redemption
    redemption_flags = (
        coupon_redempt[
            ["CAMPAIGN", "household_key"]
        ]
        .drop_duplicates()
        .assign(redeemed=1)
    )

    # 4. Validate redemption assignments
    assignment_check = redemption_flags.merge(
        campaign_targets,
        on=["CAMPAIGN", "household_key"],
        how="left",
        indicator=True,
    )

    unmatched = (
        assignment_check["_merge"] == "left_only"
    ).sum()

    if unmatched > 0:
        raise ValueError(
            f"Unmatched redemptions found: {unmatched}"
        )

    # 5. Combine campaign and redemption data
    participation = (
        campaign_targets
        .merge(
            campaign_desc[
                ["CAMPAIGN", "DESCRIPTION"]
            ],
            on="CAMPAIGN",
            how="left",
            validate="many_to_one",
        )
        .merge(
            redemption_flags,
            on=["CAMPAIGN", "household_key"],
            how="left",
            validate="one_to_one",
        )
    )

    # 6. Create redemption indicator
    participation["redeemed"] = (
        participation["redeemed"]
        .fillna(0)
        .astype(int)
    )

    # 7. Validate final dataset
    if participation["DESCRIPTION"].isna().any():
        raise ValueError("Missing campaign descriptions.")

    if not participation["redeemed"].isin([0, 1]).all():
        raise ValueError("Invalid redemption indicators.")

    if participation.duplicated(
        ["CAMPAIGN", "household_key"]
    ).any():
        raise ValueError("Duplicate campaign assignments.")

    return participation


def summarize_campaign_types(
    participation: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate redemption performance by campaign type."""

    required = {
        "CAMPAIGN",
        "household_key",
        "DESCRIPTION",
        "redeemed",
    }

    missing = required - set(participation.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # 1. Aggregate assignments by campaign type
    summary = (
        participation
        .groupby("DESCRIPTION", as_index=False)
        .agg(
            campaign_count=("CAMPAIGN", "nunique"),
            targeted_assignments=("household_key", "size"),
            unique_households=("household_key", "nunique"),
            redeemed_assignments=("redeemed", "sum"),
        )
    )

    # 2. Calculate redemption rate
    summary["redemption_rate_pct"] = (
        summary["redeemed_assignments"]
        / summary["targeted_assignments"]
        * 100
    )

    # 3. Sort by redemption performance
    summary = summary.sort_values(
        "redemption_rate_pct",
        ascending=False,
    ).reset_index(drop=True)

    return summary


def summarize_campaign_spending(
    campaign_spending: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize campaign spending by redemption status."""

    required = {
        "DESCRIPTION",
        "redeemed",
        "household_key",
        "campaign_spending",
        "campaign_baskets",
    }

    missing = required - set(campaign_spending.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if campaign_spending.empty:
        raise ValueError("Campaign spending data is empty.")

    # Calculate spending statistics
    summary = (
        campaign_spending
        .groupby(
            ["DESCRIPTION", "redeemed"],
            as_index=False,
        )
        .agg(
            assignments=("household_key", "size"),
            avg_spending=("campaign_spending", "mean"),
            median_spending=("campaign_spending", "median"),
            avg_baskets=("campaign_baskets", "mean"),
        )
    )

    return summary
