"""Retail intelligence API. All dashboard operations use historical local artifacts."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import store
from api.routes import campaigns, churn, crm, forecasting, recommendations

app = FastAPI(
    title="Dunnhumby Retail Intelligence",
    version="1.0",
    description="Historical customer, campaign, product and demand analysis.",
)
for module in (crm, churn, campaigns, recommendations, forecasting):
    app.include_router(module.router)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError):
    """Report the field and reason without echoing an invalid request body."""
    errors = [{"loc": error["loc"], "msg": error["msg"]} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(FileNotFoundError)
async def missing_artifact(request: Request, exc: FileNotFoundError):
    logging.getLogger(__name__).warning(
        "Missing artifact on %s: %s", request.url.path, exc.filename
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": f"The file {Path(exc.filename).name if exc.filename else 'for this result'} is missing. Check the artifact list in the README."
        },
    )


@app.exception_handler(Exception)
async def unavailable(request: Request, exc: Exception):
    logging.getLogger(__name__).exception("Could not serve %s", request.url.path)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "The saved data or model could not be read. Check the API log."
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict[str, dict]:
    return {
        "crm": store.metadata(store.MODELS / "crm/crm_model_metadata.json"),
        "segmentation": store.metadata(
            store.MODELS / "segmentation/segmentation_metadata.json"
        ),
        "churn": store.metadata(store.MODELS / "churn/churn_model_metadata.json"),
        "campaigns": store.metadata(
            store.DATA / "campaign/campaign_analysis_metadata.json"
        ),
        "recommendations": store.metadata(
            store.DATA / "recommendation/evaluation_metadata.json"
        ),
        "forecasting": store.metadata(store.MODELS / "forecasting/metadata.json"),
    }
