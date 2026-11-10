
import numpy as np
import pandas as pd

from lifetimes import BetaGeoFitter
from scipy.special import hyp2f1


def fit_bgnbd(
    data: pd.DataFrame,
    frequency_col: str = "frequency_cal",
    recency_col: str = "recency_cal",
    age_col: str = "T_cal",
    penalizer_coef: float = 0.01,
    tol: float | None = None,
    maxiter: int | None = None,
) -> BetaGeoFitter:
    """Train a BG-NBD model using customer purchase history."""

    required = [frequency_col, recency_col, age_col]

    if data.empty or data[required].isna().any().any():
        raise ValueError("Missing or empty BG-NBD training data.")

    frequency = data[frequency_col].to_numpy(dtype=float)
    recency = data[recency_col].to_numpy(dtype=float)
    age = data[age_col].to_numpy(dtype=float)

    if not (
        np.isfinite(frequency).all()
        and np.isfinite(recency).all()
        and np.isfinite(age).all()
        and (frequency >= 0).all()
        and (recency >= 0).all()
        and (age > 0).all()
        and (recency <= age).all()
    ):
        raise ValueError("Invalid BG-NBD customer features.")

    # Initialize model
    model = BetaGeoFitter(
        penalizer_coef=penalizer_coef
    )

    fit_kwargs = {}

    if tol is not None:
        fit_kwargs["tol"] = tol

    if maxiter is not None:
        fit_kwargs["maxiter"] = maxiter

    # Train model
    model.fit(
        frequency=frequency,
        recency=recency,
        T=age,
        **fit_kwargs,
    )

    # Validate parameters
    if not np.isfinite(
        model.params_.to_numpy(dtype=float)
    ).all():
        raise ValueError("BG-NBD parameters are not finite.")

    return model


def predict_bgnbd(
    model: BetaGeoFitter,
    data: pd.DataFrame,
    horizon: int = 90,
    frequency_col: str = "frequency_cal",
    recency_col: str = "recency_cal",
    age_col: str = "T_cal",
) -> tuple[np.ndarray, int]:
    """Predict future purchases and correct numerical issues."""

    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")

    required = [frequency_col, recency_col, age_col]

    if data.empty or data[required].isna().any().any():
        raise ValueError("Missing or empty prediction data.")

    frequency = data[frequency_col].to_numpy(dtype=float)
    recency = data[recency_col].to_numpy(dtype=float)
    age = data[age_col].to_numpy(dtype=float)

    if not (
        np.isfinite(frequency).all()
        and np.isfinite(recency).all()
        and np.isfinite(age).all()
        and (frequency >= 0).all()
        and (recency >= 0).all()
        and (age > 0).all()
        and (recency <= age).all()
    ):
        raise ValueError("Invalid prediction features.")

    # 1. Generate BG-NBD predictions
    with np.errstate(invalid="ignore"):
        predictions = np.asarray(
            model.conditional_expected_number_of_purchases_up_to_time(
                t=horizon,
                frequency=frequency,
                recency=recency,
                T=age,
            ),
            dtype=float,
        ).reshape(-1)

    # 2. Detect invalid predictions
    invalid = ~np.isfinite(predictions)
    corrected_count = int(invalid.sum())

    if corrected_count > 0:

        # Only correct customers with no repeat purchases
        if not np.all(frequency[invalid] == 0):
            raise ValueError(
                "Invalid predictions for repeat customers."
            )

        # Get fitted parameters
        r, alpha, a, b = [
            float(model.params_[param])
            for param in ["r", "alpha", "a", "b"]
        ]

        if np.isclose(a, 1.0):
            raise ValueError("Unstable BG-NBD parameter a.")

        T = age[invalid]
        z = horizon / (alpha + T + horizon)

        # 3. Numerical correction
        hypergeom = hyp2f1(
            r,
            b,
            a + b - 1,
            z,
        )

        corrected = (
            (a + b - 1) / (a - 1)
            * (
                1
                - hypergeom
                * (
                    (alpha + T)
                    / (alpha + T + horizon)
                ) ** r
            )
        )

        if not (
            np.isfinite(corrected).all()
            and (corrected >= 0).all()
        ):
            raise ValueError("Numerical correction failed.")

        predictions[invalid] = corrected

    # 4. Final validation
    if not (
        np.isfinite(predictions).all()
        and (predictions >= 0).all()
    ):
        raise ValueError("Invalid BG-NBD predictions.")

    return predictions, corrected_count
