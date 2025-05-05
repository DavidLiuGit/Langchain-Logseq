import logging
from typing import Any


def _enable_logging(log_level: int = logging.INFO):
    logging.basicConfig(
        level=log_level,  # Set to DEBUG level to see debug logs
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _print_document_count(documents_retrieved: list[Any], query: str):
    print(f"# Documents retrieved={len(documents_retrieved)}, for Q: {query}")
