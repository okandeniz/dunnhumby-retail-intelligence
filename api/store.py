"""Read trusted local artifacts once per API worker."""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
MODELS = ROOT / "models"


@lru_cache(maxsize=48)
def csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Empty artifact")
    return frame


@lru_cache(maxsize=16)
def metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=24)
def artifact(path: Path):
    return joblib.load(path)


@lru_cache(maxsize=8)
def matrix(path: Path):
    return load_npz(path).tocsr()


@lru_cache(maxsize=2)
def factors(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as bundle:
        return dict(bundle)


def records(frame: pd.DataFrame) -> list[dict]:
    """Convert numpy values and missing observations to JSON values."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def select(
    frame: pd.DataFrame, column: str, value: int | str, label: str
) -> pd.DataFrame:
    found = frame.loc[frame[column].eq(value)]
    if found.empty:
        raise HTTPException(404, f"{label} was not found.")
    return found
