
import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_campaign_redemption_gee(
    participation: pd.DataFrame,
) -> tuple[object, pd.DataFrame]:
    """Fit a GEE logistic model for coupon redemption."""

    # 1. Validate required columns
    required = {
        "household_key",
        "DESCRIPTION",
        "redeemed",
    }

    missing = required - set(participation.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if participation.empty:
        raise ValueError("Campaign participation data is empty.")

    if participation[list(required)].isna().any().any():
        raise ValueError("Missing model input values.")

    if not participation["redeemed"].isin([0, 1]).all():
        raise ValueError("Redemption must be binary.")

    if "TypeB" not in participation["DESCRIPTION"].unique():
        raise ValueError("Reference campaign TypeB is missing.")

    if participation["DESCRIPTION"].nunique() < 2:
        raise ValueError("At least two campaign types are required.")

    # 2. Define GEE model
    gee_model = sm.GEE.from_formula(
        formula=(
            'redeemed ~ C(DESCRIPTION, '
            'Treatment(reference="TypeB"))'
        ),
        groups="household_key",
        data=participation,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )

    # 3. Fit model
    gee_result = gee_model.fit()

    # 4. Calculate odds ratios and confidence intervals
    confidence_intervals = gee_result.conf_int()

    gee_summary = pd.DataFrame({
        "odds_ratio": np.exp(gee_result.params),
        "ci_lower": np.exp(confidence_intervals[0]),
        "ci_upper": np.exp(confidence_intervals[1]),
        "p_value": gee_result.pvalues,
    })

    # 5. Validate model results
    if not np.isfinite(
        gee_summary.to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite GEE results found.")

    if not gee_result.converged:
        raise RuntimeError("GEE model did not converge.")

    return gee_result, gee_summary


def fit_campaign_spending_gee(
    campaign_spending: pd.DataFrame,
    campaign_desc: pd.DataFrame,
) -> tuple[object, pd.DataFrame]:
    """Fit a GEE model for campaign-period spending."""

    # 1. Validate required columns
    spending_cols = {
        "CAMPAIGN",
        "household_key",
        "DESCRIPTION",
        "redeemed",
        "campaign_spending",
    }

    campaign_cols = {
        "CAMPAIGN",
        "START_DAY",
        "END_DAY",
    }

    for data, columns in [
        (campaign_spending, spending_cols),
        (campaign_desc, campaign_cols),
    ]:
        missing = columns - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

    if campaign_spending.empty:
        raise ValueError("Campaign spending data is empty.")

    if not campaign_desc["CAMPAIGN"].is_unique:
        raise ValueError("Duplicate campaign definitions.")

    if campaign_spending.duplicated(
        ["CAMPAIGN", "household_key"]
    ).any():
        raise ValueError("Duplicate campaign assignments.")

    # 2. Add campaign dates
    data = campaign_spending.merge(
        campaign_desc[
            ["CAMPAIGN", "START_DAY", "END_DAY"]
        ],
        on="CAMPAIGN",
        how="left",
        validate="many_to_one",
    )

    required_values = [
        "household_key",
        "DESCRIPTION",
        "redeemed",
        "campaign_spending",
        "START_DAY",
        "END_DAY",
    ]

    if data[required_values].isna().any().any():
        raise ValueError("Missing GEE model inputs.")

    if not data["redeemed"].isin([0, 1]).all():
        raise ValueError("Redemption must be binary.")

    if "TypeB" not in data["DESCRIPTION"].unique():
        raise ValueError("Reference campaign TypeB is missing.")

    numeric_cols = [
        "campaign_spending",
        "START_DAY",
        "END_DAY",
    ]

    data[numeric_cols] = data[numeric_cols].apply(
        pd.to_numeric,
        errors="raise",
    )

    if not np.isfinite(
        data[numeric_cols].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite campaign values.")

    if data["campaign_spending"].lt(0).any():
        raise ValueError("Negative spending found.")

    # 3. Calculate campaign duration
    data["campaign_duration"] = (
        data["END_DAY"] - data["START_DAY"] + 1
    )

    if data["campaign_duration"].le(0).any():
        raise ValueError("Invalid campaign duration.")

    # 4. Apply log transformation
    data["log_spending"] = np.log1p(
        data["campaign_spending"]
    )

    # 5. Build GEE spending model
    gee_model = sm.GEE.from_formula(
        formula=(
            'log_spending ~ redeemed * '
            'C(DESCRIPTION, Treatment(reference="TypeB")) '
            '+ campaign_duration'
        ),
        groups="household_key",
        data=data,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )

    # 6. Fit model
    gee_result = gee_model.fit()

    if not gee_result.converged:
        raise RuntimeError("GEE spending model did not converge.")

    # 7. Extract coefficients and confidence intervals
    confidence_intervals = gee_result.conf_int()

    gee_summary = pd.DataFrame({
        "coefficient": gee_result.params,
        "ci_lower": confidence_intervals[0],
        "ci_upper": confidence_intervals[1],
        "p_value": gee_result.pvalues,
    })

    if not np.isfinite(
        gee_summary.to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite GEE results.")

    return gee_result, gee_summary
