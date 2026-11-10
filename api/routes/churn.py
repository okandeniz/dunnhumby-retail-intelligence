from functools import lru_cache

import pandas as pd
from fastapi import APIRouter
from pydantic import ValidationError

from api import store
from api.schemas import EntityID, Features, Prediction, Sample
from src.models.churn import FEATURE_COLUMNS, predict_churn

router = APIRouter(prefix="/churn", tags=["Churn"])


@lru_cache(maxsize=1)
def classifier():
    directory = store.MODELS / "churn"
    model = store.artifact(directory / "churn_pipeline.joblib")
    meta = store.metadata(directory / "churn_model_metadata.json")
    if (
        list(model.feature_names_in_) != meta["features"]
        or meta["features"] != FEATURE_COLUMNS
    ):
        raise ValueError("Churn feature mismatch")
    return model, meta["threshold"]


def feature_rows() -> pd.DataFrame:
    return store.csv(store.DATA / "churn/churn_test_features.csv")


def score(features: Features) -> Prediction:
    model, threshold = classifier()
    result = predict_churn(
        model, pd.DataFrame([features.model_dump()]), threshold
    ).iloc[0]
    return Prediction(
        status="Churn" if result.predicted_churn else "No Churn",
        probability=result.churn_probability,
        threshold=threshold,
    )


def sample_from(frame: pd.DataFrame) -> Sample:
    row = store.records(frame)[0]
    try:
        return Sample(
            customer_id=row["household_key"],
            features=Features(**{key: row[key] for key in FEATURE_COLUMNS}),
        )
    except ValidationError as exc:
        raise ValueError("Invalid saved churn input") from exc


@router.get("/sample")
def random_sample() -> Sample:
    """Draw one test household with all six model inputs."""
    return sample_from(feature_rows().sample(n=1))


@router.post("/predict")
def manual_prediction(features: Features) -> Prediction:
    return score(features)


@router.get("/customers/{customer_id}")
def customer_prediction(customer_id: EntityID) -> Prediction:
    """Score the historical feature row and attach the separately observed outcome."""
    sample = sample_from(
        store.select(feature_rows(), "household_key", customer_id, "Customer")
    )
    prediction = score(sample.features)
    history = store.csv(store.DATA / "churn/churn_test_predictions.csv")
    actual = store.select(history, "household_key", customer_id, "Test result").iloc[0]
    prediction.customer_id = customer_id
    prediction.observed = "Churn" if actual.churn_90d else "No Churn"
    return prediction
