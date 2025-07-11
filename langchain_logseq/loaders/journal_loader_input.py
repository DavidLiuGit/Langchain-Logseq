from datetime import datetime, date
from typing import Annotated

from pydantic import BaseModel, Field, AfterValidator, computed_field, PrivateAttr, model_validator
    


def _validate_date_format(value: str) -> str:
    """
    Normalize date string to ISO format (YYYY-MM-DD).
    Accepts dates like '2023-3-1' and converts to '2023-03-01'.
    """
    try:
        _parse_date(value)
        return value
    except ValueError:
        raise ValueError(f"Invalid date: '{value}'. Expecting ISO-8601 format: YYYY-MM-DD")


def _parse_date(date_str: str) -> date:
    """
    Parse a date string into a datetime.date object.
    """
    return datetime.strptime(date_str, '%Y-%m-%d').date()



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
        AfterValidator(_validate_date_format),
    ]
    journal_end_date: Annotated[
        str,
        Field(
            description="The end date of the journal to load, in YYYY-MM-DD format.",
            examples=["2023-01-01", "2025-06-09"],
        ),
        AfterValidator(_validate_date_format),
    ]
    max_char_length: Annotated[
        int,
        Field(
            description="The maximum number of characters to include in a single `Document`.",
            examples=[8196, 2000],
            default=1024 * 8,
        ),
    ] = 1024 * 8
    enable_splitting: Annotated [
        bool,
        Field(
            description="Whether to split the journal file into multiple `Document`s.",
            examples=[True, False],
            default=True,
        ),
    ] = True


    # Private attributes that won't be included in model_dump
    _start_date: date = PrivateAttr()
    _end_date: date = PrivateAttr()

    @model_validator(mode='after')
    def _parse_dates(self) -> 'LogseqJournalLoaderInput':
        """Parse date strings into date objects after validation."""
        self._start_date = _parse_date(self.journal_start_date)
        self._end_date = _parse_date(self.journal_end_date)
        return self

    @computed_field
    @property
    def start_date(self) -> date:
        """Get `journal_start_date` as a date object."""
        return self._start_date

    @computed_field
    @property
    def end_date(self) -> date:
        """Get `journal_end_date` as a date object."""
        return self._end_date


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
