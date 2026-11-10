
from sqlalchemy import text

from src.data.database import get_engine


def main() -> None:
    """Test the MySQL database connection."""

    engine = get_engine()

    query = text(
        """
        SELECT
            VERSION() AS mysql_version,
            DATABASE() AS database_name
        """
    )

    with engine.connect() as connection:
        result = connection.execute(query).mappings().one()

    print("MySQL Connection: SUCCESS")
    print(f"Version: {result['mysql_version']}")
    print(f"Database: {result['database_name']}")


if __name__ == "__main__":
    main()
