from typing import Annotated

from pydantic import BaseModel, Field

class LogseqJournalDocumentMetadata(BaseModel):
    """
    Metadata for a Logseq journal `Document`.
    """

    journal_date: Annotated[
        str,
        Field(
            description="The date of the journal entry, in YYYY-MM-DD format.",
            examples=["2023-01-01", "2025-06-09"],
        )
    ]

    journal_tags: Annotated[
        list[str],
        Field(
            description="The tags associated with the journal entry.",
            examples=[["tag1", "tag2"], ["tag3"]],
        )
    ]

    journal_char_count: Annotated[
        int,
        Field(
            description="The number of characters in the journal entry.",
        )
    ]
