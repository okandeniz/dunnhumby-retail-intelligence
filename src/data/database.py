import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

# The decorator is used to cache the returned result.
@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and reuse a MySQL database engine."""

    url = URL.create(
        drivername="mysql+pymysql",
        username=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        database=os.environ["MYSQL_DATABASE"],
        query={"charset": "utf8mb4"},
    )

    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
    )

    return engine