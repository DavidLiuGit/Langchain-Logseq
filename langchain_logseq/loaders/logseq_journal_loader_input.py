from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, AfterValidator
    
    
def _validate_date_fields(value: str):
    """Validate format of date fields."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format.")


class LogseqJournalLoaderInput(BaseModel):
    """
    Input for a Logseq journal `Document` loader, to invoke a load.
    """
    journal_start_date: Annotated[
        str,
        Field(
            description="The start date of the journal to load, in YYYY-MM-DD format.",
            examples=["2023-01-01", "2025-06-09"],
        ),
        AfterValidator(_validate_date_fields),
    ]
    journal_end_date: Annotated[
        str,
        Field(
            description="The end date of the journal to load, in YYYY-MM-DD format.",
            examples=["2023-01-01", "2025-06-09"],
        ),
        AfterValidator(_validate_date_fields),
    ]
    max_char_length: Annotated[
        int,
        Field(
            description="The maximum number of characters to include in a single `Document`.",
            examples=[8196, 2000],
            default=1024 * 8,
        ),
    ] = 1024 * 8
