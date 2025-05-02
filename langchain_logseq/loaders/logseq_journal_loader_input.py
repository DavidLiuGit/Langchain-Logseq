from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, AfterValidator
    
    
def _validate_date_fields(value: str):
    """Validate format of date fields."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return value
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


# debugging only
if __name__ == '__main__':
    from pprint import pprint
    pprint(LogseqJournalLoaderInput.model_json_schema())

    example = LogseqJournalLoaderInput(
        journal_start_date="2023-01-01",
        journal_end_date="2023-01-02",
        max_char_length=1024 * 4,
    )
    print(example.model_dump())
