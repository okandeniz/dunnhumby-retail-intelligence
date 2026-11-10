
from pathlib import Path

import pandas as pd
from sqlalchemy import text
import json

from src.data.database import get_engine

from src.data.campaigns import load_campaign_spending
from src.features.campaigns import (
    build_campaign_participation,
    summarize_campaign_types,
    summarize_campaign_spending,
)
from src.statistics.campaign_tests import (
    analyze_redemption_association,
    analyze_campaign_spending,
)
from src.statistics.campaign_models import (
    fit_campaign_redemption_gee,
    fit_campaign_spending_gee,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "campaign"
)


def main() -> None:
    """Run the statistics and campaign analysis pipeline."""

    print("CAMPAIGN ANALYSIS PIPELINE")
    print("-" * 40)

    # 1. Create output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Define campaign data queries
    queries = {
        "campaign_desc": "SELECT * FROM raw_campaign_desc",
        "campaign_table": "SELECT * FROM raw_campaign_table",
        "coupon_redempt": "SELECT * FROM raw_coupon_redempt",
    }

    # 3. Load campaign data from MySQL
    engine = get_engine()

    try:
        with engine.connect() as connection:
            campaign_data = {
                name: pd.read_sql_query(
                    text(query),
                    con=connection,
                )
                for name, query in queries.items()
            }
    finally:
        engine.dispose()

    campaign_desc = campaign_data["campaign_desc"]
    campaign_table = campaign_data["campaign_table"]
    coupon_redempt = campaign_data["coupon_redempt"]

    print("\nDATA LOADING COMPLETED")

    for name, data in campaign_data.items():
        print(f"{name}: {data.shape}")

    # 4. Build campaign participation dataset
    participation_df = build_campaign_participation(
        campaign_table=campaign_table,
        campaign_desc=campaign_desc,
        coupon_redempt=coupon_redempt,
    )

    # 5. Summarize campaign redemption
    campaign_summary_df = summarize_campaign_types(
        participation=participation_df,
    )

    # 6. Validate participation results
    assert len(participation_df) == 7208
    assert participation_df["redeemed"].sum() == 889

    print("\nCAMPAIGN PARTICIPATION COMPLETED")
    print("Total Assignments:", len(participation_df))
    print("Redeemed Assignments:", participation_df["redeemed"].sum())

    print("\nCAMPAIGN TYPE SUMMARY")
    print(campaign_summary_df.round(2).to_string(index=False))

    # 7. Run Chi-Square Test and calculate Cramer's V
    (
        chi_square_stats,
        observed_df,
        expected_df,
    ) = analyze_redemption_association(
        participation=participation_df,
    )

    print("\nCHI-SQUARE ANALYSIS COMPLETED")
    print("Chi-Square:", round(chi_square_stats["chi2"], 4))
    print("P-Value:", f"{chi_square_stats['p_value']:.6e}")
    print("Degrees of Freedom:", chi_square_stats["degrees_of_freedom"])
    print("Cramer's V:", round(chi_square_stats["cramers_v"], 4))

    # 8. Fit GEE Logistic Regression
    gee_redemption_result, gee_redemption_df = (
        fit_campaign_redemption_gee(
            participation=participation_df,
        )
    )

    # 9. Validate GEE model
    assert gee_redemption_result.converged
    assert len(gee_redemption_df) == 3

    print("\nGEE REDEMPTION ANALYSIS COMPLETED")
    print("Model Converged:", gee_redemption_result.converged)

    print("\nGEE REDEMPTION RESULTS")
    print(gee_redemption_df.round(4).to_string())

    print("\nCAMPAIGN REDEMPTION PIPELINE: PASSED")

    # 10. Load campaign-period spending data
    engine = get_engine()

    try:
        campaign_spending_df = load_campaign_spending(
            engine=engine,
            participation=participation_df,
        )
    finally:
        engine.dispose()

    # 11. Validate campaign spending dataset
    assert len(campaign_spending_df) == 7208

    assert campaign_spending_df[
        "campaign_spending"
    ].notna().all()

    print("\nCAMPAIGN SPENDING DATA LOADED")
    print("Assignments:", len(campaign_spending_df))

    # 12. Build spending summary
    spending_summary_df = summarize_campaign_spending(
        campaign_spending=campaign_spending_df,
    )

    assert len(spending_summary_df) == 6

    print("\nCAMPAIGN SPENDING SUMMARY")
    print(
        spending_summary_df.round(2).to_string(
            index=False
        )
    )

    # 13. Run Mann-Whitney U tests
    mann_whitney_df = analyze_campaign_spending(
        campaign_spending=campaign_spending_df,
    )

    assert len(mann_whitney_df) == 3

    print("\nMANN-WHITNEY U ANALYSIS COMPLETED")

    print(
        mann_whitney_df.round(4).to_string(
            index=False
        )
    )

    # 14. Fit GEE Spending model
    (
        gee_spending_result,
        gee_spending_df,
    ) = fit_campaign_spending_gee(
        campaign_spending=campaign_spending_df,
        campaign_desc=campaign_desc,
    )

    # 15. Validate GEE model
    assert gee_spending_result.converged
    assert len(gee_spending_df) == 7

    print("\nGEE SPENDING ANALYSIS COMPLETED")
    print("Model Converged:", gee_spending_result.converged)

    print("\nGEE SPENDING RESULTS")
    print(
        gee_spending_df.round(4).to_string()
    )

    print("\nCAMPAIGN SPENDING PIPELINE: PASSED")

     # 16. Define output files
    csv_outputs = {
        "campaign_participation.csv": participation_df,
        "campaign_type_summary.csv": campaign_summary_df,
        "campaign_spending.csv": campaign_spending_df,
        "campaign_spending_summary.csv": spending_summary_df,
        "mann_whitney_results.csv": mann_whitney_df,
    }

    # 17. Save standard CSV outputs
    for filename, dataframe in csv_outputs.items():
        dataframe.to_csv(
            PROCESSED_DIR / filename,
            index=False,
        )

    # 18. Save GEE results with parameter names
    gee_redemption_df.to_csv(
        PROCESSED_DIR / "gee_redemption_summary.csv",
        index=True,
        index_label="term",
    )

    gee_spending_df.to_csv(
        PROCESSED_DIR / "gee_spending_summary.csv",
        index=True,
        index_label="term",
    )

    # 19. Save Chi-Square frequency tables
    observed_df.to_csv(
        PROCESSED_DIR / "chi_square_observed.csv",
        index=True,
        index_label="DESCRIPTION",
    )

    expected_df.to_csv(
        PROCESSED_DIR / "chi_square_expected.csv",
        index=True,
        index_label="DESCRIPTION",
    )

    # 20. Save Chi-Square statistics
    with open(
        PROCESSED_DIR / "chi_square_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chi_square_stats,
            file,
            indent=4,
        )

    # 21. Save analysis metadata
    metadata = {
        "analysis_name": "Statistics and Campaign Analysis",
        "analysis_type": "observational",
        "total_assignments": int(len(participation_df)),
        "redeemed_assignments": int(
            participation_df["redeemed"].sum()
        ),
        "campaign_types": sorted(
            participation_df["DESCRIPTION"].unique().tolist()
        ),
        "statistical_methods": [
            "Chi-Square Test",
            "Cramer's V",
            "GEE Logistic Regression",
            "Mann-Whitney U",
            "Cliff's Delta",
            "Holm Correction",
            "GEE Gaussian Regression",
        ],
        "gee_redemption_converged": bool(
            gee_redemption_result.converged
        ),
        "gee_spending_converged": bool(
            gee_spending_result.converged
        ),
        "causal_inference": False,
    }

    with open(
        PROCESSED_DIR / "campaign_analysis_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # 22. Verify saved outputs
    expected_files = (
        list(csv_outputs)
        + [
            "gee_redemption_summary.csv",
            "gee_spending_summary.csv",
            "chi_square_observed.csv",
            "chi_square_expected.csv",
            "chi_square_results.json",
            "campaign_analysis_metadata.json",
        ]
    )

    for filename in expected_files:
        path = PROCESSED_DIR / filename

        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing or empty artifact: {path}"
            )

    # 23. Check saved spending results
    saved_spending = pd.read_csv(
        PROCESSED_DIR / "campaign_spending.csv"
    )

    pd.testing.assert_frame_equal(
        campaign_spending_df.reset_index(drop=True),
        saved_spending,
        check_dtype=False,
        rtol=1e-9,
        atol=1e-6,
    )

    print("\nARTIFACT SAVING COMPLETED")
    print("Output Directory:", PROCESSED_DIR)
    print("Saved Files:", len(expected_files))

    print("\nArtifact Validation: PASSED")
    print(
        "\nCAMPAIGN ANALYSIS PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
