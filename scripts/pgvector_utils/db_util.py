import os
from dotenv import load_dotenv

load_dotenv()


def database_url() -> str:
    url = os.getenv("PGVECTOR_URL")
    if not url:
        raise ValueError("PGVECTOR_URL environment variable not set")
    return url
