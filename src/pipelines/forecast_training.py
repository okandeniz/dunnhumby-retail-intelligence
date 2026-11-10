
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.database import get_engine

from src.data.forecasting import (
    load_forecast_product_lookup,
    load_product_demand_profile,
    select_eligible_products,
    load_daily_product_sales,
    build_daily_product_series,
)

from src.features.forecasting import (
    build_historical_features,
    select_valid_forecast_features,
    build_forecast_targets,
    split_forecast_dataset,
    prepare_forecast_model_inputs,
)

from src.models.forecasting import (
    build_selected_forecast_model,
    predict_selected_forecast_model,
    add_baseline_predictions,
)

from src.evaluation.forecasting import (
    evaluate_forecast,
    compare_baseline_models,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "forecasting"

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed" / "forecasting"
)


def main() -> None:
    """Train and evaluate the selected demand forecasting model."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("PRODUCT DEMAND FORECASTING PIPELINE")
    print("-" * 45)

    # 1. Load source data
    engine = get_engine()

    try:
        product_lookup = load_forecast_product_lookup(
            engine
        )

        product_profile = load_product_demand_profile(
            engine=engine,
            product_lookup=product_lookup,
            train_end_day=531,
        )

        eligible_products, eligible_lookup = (
            select_eligible_products(
                product_demand_profile=product_profile,
                product_lookup=product_lookup,
            )
        )

        daily_sales = load_daily_product_sales(
            engine=engine,
            eligible_products=eligible_products,
            max_day=711,
        )
    finally:
        engine.dispose()

    assert len(eligible_products) == 457

    print("DATA LOADING: PASSED")
    print("Eligible Products:", len(eligible_products))

    # 2. Prepare complete daily panel
    daily_series = build_daily_product_series(
        daily_sales=daily_sales,
        eligible_products=eligible_products,
        max_day=711,
    )

    # 3. Generate historical features
    historical_features = build_historical_features(
        daily_product_series=daily_series,
    )

    valid_features = select_valid_forecast_features(
        forecast_features=historical_features,
        min_history_days=60,
    )

    # 4. Generate 30-day targets
    _, _, model_dataset = build_forecast_targets(
        daily_product_series=daily_series,
        valid_features=valid_features,
        forecast_horizon=30,
    )

    # 5. Create time-based splits
    splits, split_summary = split_forecast_dataset(
        model_dataset=model_dataset,
        train_end_day=531,
        val_end_day=621,
        test_end_day=711,
        forecast_horizon=30,
        first_train_origin=111,
    )

    train = splits["Train"]
    validation = splits["Validation"]
    test = splits["Test"]

    assert len(train) == 6378
    assert len(validation) == 1371
    assert len(test) == 1371

    print("\nFEATURE ENGINEERING: PASSED")
    print("TIME-BASED SPLIT: PASSED")
    print(split_summary.to_string(index=False))

    # 6. Combine Train and Validation
    final_training_data = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    # 7. Prepare model inputs
    X_train, y_train = prepare_forecast_model_inputs(
        forecast_data=final_training_data,
        eligible_product_lookup=eligible_lookup,
    )

    X_test, y_test = prepare_forecast_model_inputs(
        forecast_data=test,
        eligible_product_lookup=eligible_lookup,
    )

    assert len(X_train) == 7749
    assert len(X_test) == 1371
    assert X_train.columns.equals(X_test.columns)

    # 8. Train selected model
    model = build_selected_forecast_model()

    model.fit(X_train, y_train)

    print("\nFINAL MODEL TRAINING: PASSED")
    print("Training Observations:", len(X_train))

    # 9. Generate independent test predictions
    test_predictions_array = predict_selected_forecast_model(
        model=model,
        X=X_test,
    )

    # 10. Evaluate final model
    final_metrics = evaluate_forecast(
        y_true=y_test,
        y_pred=test_predictions_array,
    )

    # 11. Evaluate both baseline methods on test
    baseline_test = add_baseline_predictions(
        forecast_data=test,
    )

    baseline_metrics = compare_baseline_models(
        baseline_data=baseline_test,
    )

    # 12. Compare final model with baselines
    final_comparison = pd.concat(
        [
            baseline_metrics,
            pd.DataFrame(
                [final_metrics],
                index=["XGBoost + Log Target"],
            ),
        ]
    ).sort_values("MAE")

    print("\nFINAL TEST MODEL COMPARISON")
    print(final_comparison.round(3))

    # 13. Validate against notebook reference
    reference_metrics = {
        "MAE": 24.872,
        "RMSE": 53.544,
        "WAPE (%)": 27.332,
        "Bias": -7.497,
        "Bias (%)": -8.239,
    }

    for metric_name, expected_value in reference_metrics.items():
        np.testing.assert_allclose(
            final_metrics[metric_name],
            expected_value,
            rtol=0,
            atol=0.15,
            err_msg=f"Notebook mismatch: {metric_name}",
        )

    print("\nNOTEBOOK TEST COMPARISON: PASSED")

    # 14. Prepare results for Plotly and FastAPI
    test_results = test[
        ["PRODUCT_ID", "DAY", "target_end_day"]
    ].reset_index(drop=True).copy()

    test_results["actual"] = y_test.to_numpy()

    test_results["xgb_prediction"] = (
        test_predictions_array
    )

    test_results["baseline_prediction"] = (
        baseline_test["pred_last_30d"].to_numpy()
    )

    test_results["baseline_60d_prediction"] = (
        baseline_test["pred_avg_60d"].to_numpy()
    )

    test_results = test_results.merge(
        eligible_lookup[
            ["PRODUCT_ID", "product_label"]
        ],
        on="PRODUCT_ID",
        how="left",
        validate="many_to_one",
    )

    test_results["forecast_period"] = (
        (test_results["DAY"] + 1).astype(str)
        + "-"
        + test_results["target_end_day"].astype(str)
    )

    test_results["error"] = (
        test_results["xgb_prediction"]
        - test_results["actual"]
    )

    test_results["absolute_error"] = (
        test_results["error"].abs()
    )

    assert len(test_results) == 1371
    assert test_results["product_label"].notna().all()

    print("TEST PREDICTION DATASET: PASSED")

    # 15. Save final model
    model_path = MODEL_DIR / "xgb_log_target.joblib"

    joblib.dump(model, model_path)

    # 16. Save data and evaluation artifacts
    eligible_lookup.to_csv(
        DATA_DIR / "eligible_product_lookup.csv",
        index=False,
    )

    daily_series.to_csv(
        DATA_DIR / "daily_product_series.csv",
        index=False,
    )

    test_results.to_csv(
        DATA_DIR / "test_predictions.csv",
        index=False,
    )

    final_comparison.to_csv(
        DATA_DIR / "final_test_comparison.csv",
    )

    split_summary.to_csv(
        DATA_DIR / "forecast_split_summary.csv",
        index=False,
    )

    # 17. Save model metadata
    metadata = {
        "model_name": "XGBoost + Log Target",
        "model_version": "xgb_log_target_v1",
        "forecast_horizon": 30,
        "products": int(len(eligible_products)),
        "training_observations": int(len(X_train)),
        "test_observations": int(len(X_test)),
        "training_target_end_day": 621,
        "last_observed_day": 711,
        "numeric_features": 10,
        "categorical_features": 3,
        "feature_names": X_train.columns.tolist(),
        "test_metrics": final_metrics,
    }

    with open(
        MODEL_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nMODEL ARTIFACTS SAVED")

    # 18. Reload and validate saved model
    loaded_model = joblib.load(model_path)

    loaded_predictions = predict_selected_forecast_model(
        model=loaded_model,
        X=X_test,
    )

    np.testing.assert_allclose(
        test_predictions_array,
        loaded_predictions,
        rtol=1e-7,
        atol=1e-6,
    )

    print("SAVED MODEL VALIDATION: PASSED")

    print(
        "\nFORECAST TRAINING PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()
