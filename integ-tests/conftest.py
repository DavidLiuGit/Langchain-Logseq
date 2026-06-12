"""
Configuration for integration tests
"""

import os
import pytest
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")


@pytest.fixture
def database_url():
    url = os.getenv("TEST_PGVECTOR_URL")
    if not url:
        pytest.skip("TEST_PGVECTOR_URL environment variable not set")
    return url
