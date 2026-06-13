from typing import Type

from pgvector.sqlalchemy import Vector
from pydantic import Field, model_validator
from sqlalchemy import Column, Index, String, text

from pgvector_template.core import (
    BaseDocument,
    BaseDocumentMetadata,
    BaseSearchClientConfig,
)
from pgvector_template.models.search import (
    SearchQuery,
    MetadataFilter,
)


class JournalDocument(BaseDocument):
    """
    Each `Corpus` is the entire entry for a given date. A corpus may consist of 1 or more chunks of `Document`s.
    Each `Corpus` has a set of metadata, and each `Document` chunk has all of those, plus more.
    """

    __abstract__ = False
    __tablename__ = "logseq_journal"

    corpus_id = Column(String(len("2025-06-09")), index=True)
    """Length of ISO date string"""
    embedding = Column(Vector())
    """Embedding vector — dimensionless to survive future model dim changes"""

    @classmethod
    def get_embedding_index(cls, table_name: str) -> Index:
        """HNSW on a cast expression — required for dimensionless vector columns.
        Update the cast dim here when switching embedding models; no table drop needed."""
        return Index(
            f"{table_name}_embedding_hnsw_idx",
            text("(embedding::vector(1536)) vector_cosine_ops"),
            postgresql_using="hnsw",
        )


class JournalCorpusMetadata(BaseDocumentMetadata):
    """Metadata schema for Logseq journal corpora. Consist of 1-or-more chunks, called `Document`s."""

    # corpus
    date_str: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date in ISO format, e.g. `2025-04-20`",
    )
    # defaults
    document_type: str = Field(default="logseq_journal")
    schema_version: str = Field(default="2026-06-02")


class JournalDocumentMetadata(JournalCorpusMetadata):
    """Metadata schema for Logseq journal `Document`s. 1-or-more `Document`s make up a corpus."""

    # chunk/document
    chunk_len: int = Field()
    """Length of the content in characters"""
    word_count: int | None = Field()
    """Length of the content in words"""
    references: list[str] = Field(default=[])
    """List of `#tag` references extracted from the entry. Note: `[[wiki-link]]` references are not currently captured."""
    anchor_ids: list[str] = Field(default=[])
    """Blocks in the document can have UUID anchors, which are referenced elsewhere. This is a list of all present"""


class JournalSearchClientConfig(BaseSearchClientConfig):
    """Configuration for the Logseq journal search client."""

    document_cls: Type[BaseDocument] = JournalDocument
    """The document type to use for the search client."""
    document_metadata_cls: Type[BaseDocumentMetadata] = JournalDocumentMetadata
    """The document metadata type to use for the search client."""
    # embedding_provider


class JournalSearchQuery(SearchQuery):
    """
    Search query for Logseq journal entries. At least one criterion is required.
    All criteria are ANDed: semantic ranking (`text`), substring filter (`keywords`),
    journal-date bounds (`date_from`/`date_to`), and arbitrary metadata (`metadata_filters`).

    Example — topic search scoped to a date range:
        text="recovering from a knee injury", date_from="2025-01-01", date_to="2025-03-31"

    Example — certain a term appears, scoped to a tag:
        keywords=["Dr. Lim", "Dr Lim"], metadata_filters=[MetadataFilter(field_name="references", condition="contains", value="health")]

    Example — everything in a month, no semantic query:
        date_from="2025-05-01", date_to="2025-05-31"
    """

    text: str | None = None
    """
    Phrase for semantic (vector) similarity search.
    Rephrase as a statement matching the content you expect to find, not as a question.
    E.g. "recovering from a knee injury", not "what did I write about my knee?".
    """

    keywords: list[str] = []
    """
    Keywords for a case-insensitive **substring** search (SQL `ILIKE '%keyword%'`).
    An entry matches if **any** keyword appears anywhere in its content (OR-combined).
    Pass multiple variants, synonyms, or spellings to increase recall — e.g.
    ["Dr. Lim", "Dr Lim"] matches either form.
    Note: substring matching means "run" also matches "running"; prefer distinctive terms.
    Do not use for fields covered by `metadata_filters` or `date_from`/`date_to`.
    """

    date_from: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description=(
            "Inclusive lower bound on the journal entry's own date (YYYY-MM-DD). "
            "Filters by the date the entry is *about*, not the database write time."
        ),
    )
    date_to: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Inclusive upper bound on the journal entry's own date (YYYY-MM-DD).",
    )

    metadata_filters: list[MetadataFilter] = Field(
        default=[],
        json_schema_extra={
            "metadata_schema": JournalDocumentMetadata.model_json_schema()
        },
    )
    """
    List of metadata conditions that must be matched (ANDed together).
    Refer to `metadata_schema` for the filterable fields and their types.
    Prefer `date_from`/`date_to` for journal-date filtering over hand-writing date_str filters here.
    """

    limit: int = Field(20, ge=3)
    """Maximum number of results to return."""

    @model_validator(mode="after")
    def ensure_criterion(self) -> "JournalSearchQuery":
        if not any(
            [
                self.text,
                self.keywords,
                self.metadata_filters,
                self.date_from,
                self.date_to,
            ]
        ):
            raise ValueError("At least one search criterion is required")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        injected = {(f.field_name, f.condition) for f in self.metadata_filters}
        extra: list[MetadataFilter] = []
        if self.date_from and ("date_str", "gte") not in injected:
            extra.append(
                MetadataFilter(
                    field_name="date_str", condition="gte", value=self.date_from
                )
            )
        if self.date_to and ("date_str", "lte") not in injected:
            extra.append(
                MetadataFilter(
                    field_name="date_str", condition="lte", value=self.date_to
                )
            )
        if extra:
            self.metadata_filters = list(self.metadata_filters) + extra
        return self
