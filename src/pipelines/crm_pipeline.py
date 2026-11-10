
from pathlib import Path

import numpy as np
from src.data.database import get_engine
from src.data.baskets import load_basket_summary
from src.features.rfm import (
    build_rfm_features,
    add_rfm_scores,
    assign_rfm_segments,
)
from src.features.cltv import (
    prepare_purchase_days,
    build_cltv_features,
)
from src.models.bgnbd import fit_bgnbd
from src.models.customer_value import (
    predict_customer_revenue,
    integrate_rfm_revenue,
    summarize_segment_forecasts,
)
import json
import joblib


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "crm"
MODEL_DIR = PROJECT_ROOT / "models" / "crm"


def main() -> None:
    """Run the CRM analytics training pipeline."""

    print("CRM ANALYTICS PIPELINE")
    print("-" * 40)

    # 1. Create output directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Connect to MySQL
    engine = get_engine()

    # 3. Load basket-level transactions
    try:
        basket_df = load_basket_summary(engine)
    finally:
        engine.dispose()

    # 4. Build RFM features
    rfm_df = build_rfm_features(
        baskets=basket_df,
        snapshot_day=712,
        lookback_days=365,
    )

    # 5. Calculate RFM scores
    rfm_df = add_rfm_scores(rfm_df)

    # 6. Assign customer segments
    rfm_df = assign_rfm_segments(rfm_df)

    # 7. Validate RFM results
    assert rfm_df["household_key"].is_unique
    assert rfm_df["segment"].notna().all()

    print("\nRFM ANALYSIS COMPLETED")
    print("Total Customers:", len(rfm_df))
    print("Total Segments:", rfm_df["segment"].nunique())

    print("\nSEGMENT DISTRIBUTION")
    print(rfm_df["segment"].value_counts().to_string())

    # 8. Prepare purchase-day data for BG-NBD
    purchase_days = prepare_purchase_days(
        baskets=basket_df,
        snapshot_day=712,
        lookback_days=275,
    )

    # 9. Build customer-level CLTV features
    cltv_df = build_cltv_features(
        purchase_days=purchase_days,
        snapshot_day=712,
    )

    # 10. Validate CLTV features
    assert cltv_df["household_key"].is_unique
    assert cltv_df.notna().all().all()
    assert cltv_df["frequency"].ge(0).all()
    assert cltv_df["recency"].le(cltv_df["T"]).all()

    print("\nCLTV FEATURE ENGINEERING COMPLETED")
    print("Training Period: Days 437-711")
    print("Training Customers:", len(cltv_df))

    print(
        "Repeat Customers:",
        cltv_df["frequency"].gt(0).sum(),
    )

    print(
        "No Repeat Customers:",
        cltv_df["frequency"].eq(0).sum(),
    )

    print("\nCLTV FEATURE SUMMARY")
    print(
        cltv_df[
            ["frequency", "recency", "T", "monetary_value"]
        ].describe().round(2).to_string()
    )

    # 11. Train the final BG-NBD model
    bgnbd_model = fit_bgnbd(
        data=cltv_df,
        frequency_col="frequency",
        recency_col="recency",
        age_col="T",
        penalizer_coef=0.01,
        tol=1e-6,
        maxiter=2000,
    )

    # 12. Display fitted model parameters
    print("\nBG-NBD MODEL TRAINING COMPLETED")
    print("Training Customers:", len(cltv_df))
    print("Penalizer Coefficient:", 0.01)

    print("\nMODEL PARAMETERS")
    print(bgnbd_model.params_.round(6).to_string())

    # 13. Validate model parameters

    assert np.isfinite(
        bgnbd_model.params_.to_numpy(dtype=float)
    ).all()

    assert (
        bgnbd_model.params_.to_numpy(dtype=float) > 0
    ).all()

    print("\nBG-NBD Parameter Validation: PASSED")

    # 14. Generate 90-day customer revenue predictions
    revenue_df, corrected_count = predict_customer_revenue(
        model=bgnbd_model,
        purchase_days=purchase_days,
        snapshot_day=712,
        horizon=90,
    )

    # 15. Validate predictions
    prediction_cols = [
        "expected_purchases_90d",
        "historical_monetary",
        "expected_revenue_90d",
    ]

    assert len(revenue_df) == len(cltv_df)
    assert revenue_df["household_key"].is_unique

    assert np.isfinite(
        revenue_df[prediction_cols].to_numpy(dtype=float)
    ).all()

    assert revenue_df[prediction_cols].ge(0).all().all()

    # 16. Display prediction results
    print("\nCUSTOMER REVENUE PREDICTION COMPLETED")

    print("Forecast Horizon: 90 Days")
    print("Forecasted Customers:", len(revenue_df))
    print("Corrected BG-NBD Predictions:", corrected_count)

    print(
        "Total Expected Revenue:",
        round(revenue_df["expected_revenue_90d"].sum(), 2),
    )

    print("\nREVENUE PREDICTION SUMMARY")

    print(
        revenue_df[prediction_cols]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nFIRST 10 CUSTOMER PREDICTIONS")

    print(
        revenue_df[
            ["household_key", *prediction_cols]
        ]
        .head(10)
        .round(2)
        .to_string(index=False)
    )

    # 17. Integrate RFM segments with revenue predictions
    crm_forecast_df = integrate_rfm_revenue(
        rfm=rfm_df,
        revenue=revenue_df,
        horizon=90,
    )

    # 18. Create segment-level revenue summary
    segment_summary_df = summarize_segment_forecasts(
        crm_forecasts=crm_forecast_df,
        horizon=90,
    )

    # 19. Validate customer coverage
    total_customers = len(crm_forecast_df)

    forecasted_customers = (
        crm_forecast_df["expected_revenue_90d"]
        .notna()
        .sum()
    )

    customers_without_forecast = (
        crm_forecast_df["expected_revenue_90d"]
        .isna()
        .sum()
    )

    assert total_customers == len(rfm_df)
    assert crm_forecast_df["household_key"].is_unique

    assert forecasted_customers == len(revenue_df)

    assert (
        forecasted_customers + customers_without_forecast
        == total_customers
    )

    assert (
        segment_summary_df["customer_count"].sum()
        == total_customers
    )

    # 20. Display results
    print("\nCRM INTEGRATION COMPLETED")

    print("Total Customers:", total_customers)
    print("Customers With Forecasts:", forecasted_customers)
    print("Customers Without Forecasts:", customers_without_forecast)

    print(
        "Forecast Coverage (%):",
        round(forecasted_customers / total_customers * 100, 2),
    )

    print("\nSEGMENT REVENUE SUMMARY")
    print(segment_summary_df.round(2).to_string(index=False))

    print("\nCRM Integration Validation: PASSED")

    # 21. Define artifact paths
    artifact_paths = {
        "rfm": PROCESSED_DIR / "rfm_segments.csv",
        "cltv": PROCESSED_DIR / "cltv_training_features.csv",
        "revenue": PROCESSED_DIR / "customer_revenue_predictions.csv",
        "crm": PROCESSED_DIR / "crm_customer_forecasts.csv",
        "segment_summary": (
            PROCESSED_DIR / "crm_segment_forecast_summary.csv"
        ),
        "model": MODEL_DIR / "bgnbd_model.joblib",
        "metadata": MODEL_DIR / "crm_model_metadata.json",
    }

    # 22. Save processed datasets
    rfm_df.to_csv(
        artifact_paths["rfm"],
        index=False,
    )

    cltv_df.to_csv(
        artifact_paths["cltv"],
        index=False,
    )

    revenue_df.to_csv(
        artifact_paths["revenue"],
        index=False,
    )

    crm_forecast_df.to_csv(
        artifact_paths["crm"],
        index=False,
    )

    segment_summary_df.to_csv(
        artifact_paths["segment_summary"],
        index=False,
    )

    # 23. Save trained BG-NBD model
    joblib.dump(
        bgnbd_model,
        artifact_paths["model"],
    )

    # 24. Prepare model metadata
    metadata = {
        "model_name": "BG-NBD",
        "model_type": "Customer Purchase Prediction",
        "snapshot_day": 712,
        "rfm_lookback_days": 365,
        "training_days": 275,
        "training_start_day": 437,
        "training_end_day": 711,
        "forecast_horizon_days": 90,
        "penalizer_coef": 0.01,
        "training_customers": int(len(cltv_df)),
        "total_customers": int(total_customers),
        "forecasted_customers": int(forecasted_customers),
        "customers_without_forecast": int(
            customers_without_forecast
        ),
        "corrected_predictions": int(corrected_count),
        "total_expected_revenue": float(
            revenue_df["expected_revenue_90d"].sum()
        ),
        "monetary_method": "Historical Monetary",
        "model_parameters": {
            key: float(value)
            for key, value in bgnbd_model.params_.items()
        },
    }

    # 25. Save metadata as JSON
    with open(
        artifact_paths["metadata"],
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # 26. Reload and validate saved model
    loaded_model = joblib.load(
        artifact_paths["model"]
    )

    np.testing.assert_allclose(
        bgnbd_model.params_.to_numpy(dtype=float),
        loaded_model.params_.to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )

    # 27. Verify output files
    for name, path in artifact_paths.items():
        assert path.is_file(), f"Artifact missing: {path}"

    print("\nARTIFACT SAVING COMPLETED")
    print("-" * 40)

    for name, path in artifact_paths.items():
        print(f"{name}: {path}")

    print("\nModel Reload Check: PASSED")
    print("Artifact Validation: PASSED")

    print("\nCRM PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
