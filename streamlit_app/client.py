"""Small HTTP client. Cache historical reads, keep random draws fresh."""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BASE = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


class ApiError(Exception):
    """A message that can be shown in the dashboard."""


def request(path: str, *, params: dict | None = None, body: dict | None = None):
    try:
        response = requests.request(
            "POST" if body is not None else "GET",
            BASE + path,
            params=params,
            json=body,
            timeout=(3, 20),
        )
    except requests.RequestException as exc:
        raise ApiError(
            "We could not reach the API. Check that the API is running and try again."
        ) from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiError(
            "The server returned an unreadable reply. Check the API URL."
        ) from exc
    if 200 <= response.status_code < 300:
        return payload
    if response.status_code in (404, 422, 503):
        detail = payload.get("detail")
        if isinstance(detail, str):
            raise ApiError(detail)
        if isinstance(detail, list) and detail:
            message = detail[0].get("msg", "Check the input values.")
            raise ApiError(message.removeprefix("Value error, "))
    raise ApiError("The API could not complete this request. Please try again.")


@st.cache_data(ttl=600, max_entries=256, show_spinner=False)
def read(path: str, **params):
    return request(path, params=params)
