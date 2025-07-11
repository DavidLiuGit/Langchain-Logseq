import os
from dotenv import load_dotenv

load_dotenv()


def database_url():
    """Get database URL from environment variables"""
    username = os.getenv("TEST_PGVECTOR_USERNAME")
    password = os.getenv("TEST_PGVECTOR_PASSWORD")
    host = os.getenv("TEST_PGVECTOR_HOST")
    port = os.getenv("TEST_PGVECTOR_PORT")
    db = os.getenv("TEST_PGVECTOR_DB")
    db_url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db}"
    if not db_url:
        raise ValueError("Database URL not set in environment variables")
    return db_url
