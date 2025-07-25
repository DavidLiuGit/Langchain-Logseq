"""
Configuration for integration tests
"""

import os
import pytest
from dotenv import load_dotenv
from pathlib import Path

# Get the directory where conftest.py is located, and load its .env file
INTEG_TEST_DIR = Path(__file__).parent.absolute()
load_dotenv(INTEG_TEST_DIR / ".env")


# Fixture for database URL - will be implemented by the user
@pytest.fixture
def database_url():
    """Get database URL from environment variables"""
    username = os.getenv("TEST_PGVECTOR_USERNAME")
    password = os.getenv("TEST_PGVECTOR_PASSWORD")
    host = os.getenv("TEST_PGVECTOR_HOST")
    port = os.getenv("TEST_PGVECTOR_PORT")
    db = os.getenv("TEST_PGVECTOR_DB")
    db_url = f"postgresql+psycopg://{username}:{password}@{host}:{port}/{db}"
    if not db_url:
        pytest.skip("TEST_DATABASE_URL environment variable not set")
    return db_url

# import more fixtures from utils
from utils.pgvector_document_service import *
