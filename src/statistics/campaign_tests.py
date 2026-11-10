
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def analyze_redemption_association(
    participation: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Run Chi-Square Test and calculate Cramer's V."""

    # 1. Validate required columns
    required = {"DESCRIPTION", "redeemed"}
    missing = required - set(participation.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if participation.empty:
        raise ValueError("Campaign participation data is empty.")

    if participation[list(required)].isna().any().any():
        raise ValueError("Missing campaign or redemption values.")

    if not participation["redeemed"].isin([0, 1]).all():
        raise ValueError("Redemption must contain binary values.")

    # 2. Build contingency table
    contingency_table = pd.crosstab(
        participation["DESCRIPTION"],
        participation["redeemed"],
    )

    contingency_table = contingency_table.reindex(
        columns=[0, 1],
        fill_value=0,
    )

    if (
        len(contingency_table) < 2
        or (contingency_table.sum(axis=0) == 0).any()
    ):
        raise ValueError(
            "At least two campaign types and both outcomes are required."
        )

    # 3. Run Chi-Square Test
    chi2, p_value, dof, expected = chi2_contingency(
        contingency_table
    )

    # 4. Calculate Cramer's V
    n = int(contingency_table.to_numpy().sum())

    min_dimension = min(
        contingency_table.shape[0] - 1,
        contingency_table.shape[1] - 1,
    )

    cramers_v = np.sqrt(
        chi2 / (n * min_dimension)
    )

    # 5. Prepare expected frequencies
    expected_df = pd.DataFrame(
        expected,
        index=contingency_table.index,
        columns=["Not Redeemed", "Redeemed"],
    )

    # 6. Prepare statistics
    statistics = {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(cramers_v),
        "min_expected_frequency": float(expected.min()),
        "observations": n,
    }

    # 7. Format observed frequencies
    observed_df = contingency_table.rename(
        columns={
            0: "Not Redeemed",
            1: "Redeemed",
        }
    )

    return statistics, observed_df, expected_df


def analyze_campaign_spending(
    campaign_spending: pd.DataFrame,
) -> pd.DataFrame:
    """Compare campaign spending using Mann-Whitney U tests."""

    # 1. Validate required columns
    required = {
        "DESCRIPTION",
        "redeemed",
        "campaign_spending",
    }

    missing = required - set(campaign_spending.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if campaign_spending.empty:
        raise ValueError("Campaign spending data is empty.")

    if campaign_spending[list(required)].isna().any().any():
        raise ValueError("Missing campaign spending values.")

    if not campaign_spending["redeemed"].isin([0, 1]).all():
        raise ValueError("Redemption must be binary.")

    spending = campaign_spending[
        "campaign_spending"
    ].to_numpy(dtype=float)

    if not np.isfinite(spending).all():
        raise ValueError("Non-finite spending values.")

    # 2. Compare redeemed and non-redeemed assignments
    test_results = []

    for campaign_type, group in campaign_spending.groupby(
        "DESCRIPTION"
    ):

        redeemed = group.loc[
            group["redeemed"].eq(1),
            "campaign_spending",
        ].to_numpy(dtype=float)

        not_redeemed = group.loc[
            group["redeemed"].eq(0),
            "campaign_spending",
        ].to_numpy(dtype=float)

        if len(redeemed) == 0 or len(not_redeemed) == 0:
            raise ValueError(
                f"Missing comparison group: {campaign_type}"
            )

        # 3. Mann-Whitney U Test
        statistic, p_value = mannwhitneyu(
            redeemed,
            not_redeemed,
            alternative="two-sided",
            method="asymptotic",
        )

        # 4. Calculate Cliff's Delta
        cliffs_delta = (
            2 * statistic
            / (len(redeemed) * len(not_redeemed))
            - 1
        )

        # 5. Store campaign-level results
        test_results.append({
            "campaign_type": campaign_type,
            "redeemed_n": len(redeemed),
            "not_redeemed_n": len(not_redeemed),
            "median_redeemed": float(np.median(redeemed)),
            "median_not_redeemed": float(
                np.median(not_redeemed)
            ),
            "u_statistic": float(statistic),
            "p_value": float(p_value),
            "cliffs_delta": float(cliffs_delta),
        })

    # 6. Create result DataFrame
    results = pd.DataFrame(test_results)

    # 7. Apply Holm multiple-testing correction
    results["p_adjusted"] = multipletests(
        results["p_value"],
        method="holm",
    )[1]

    return results

