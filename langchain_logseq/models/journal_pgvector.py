from typing import Type

from pgvector.sqlalchemy import Vector
from pydantic import Field
from sqlalchemy import Column, String

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
    embedding = Column(Vector(1024))
    """Embedding vector"""


class JournalCorpusMetadata(BaseDocumentMetadata):
    """Metadata schema for Logseq journal corpora. Consist of 1-or-more chunks, called `Document`s."""

    # corpus
    date_str: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date in ISO format, e.g. `2025-04-20`"
    )

    # defaults
    document_type: str = Field(default="logseq_journal")
    schema_version: str = Field(default="2025-07-10")


class JournalDocumentMetadata(JournalCorpusMetadata):
    """Metadata schema for Logseq journal `Document`s. 1-or-more `Document`s make up a corpus."""

    # chunk/document
    chunk_len: int = Field()
    """Length of the content in characters"""
    word_count: int | None = Field()
    """Length of the content in words"""
    references: list[str] = Field(default=[])
    """List of references to other Logseq documents, or journal dates"""
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
    Standardized search query structure, specifically for searching Logseq `JournalDocument`s.
    At least 1 search criterion is required (text, keywords, metadata_filters), but multiple are allowed.
    Types are the same as in `SearchQuery`.
    Descriptions are customized to better suit Logseq `JournalDocument`'s.
    """

    text: str | None = None
    """
    String to match against using in a semantic search, i.e. using vector distance.
    Instead of passing in a question, rephrase the question to be a string/phrase matching closer
    to the content expected to be found.
    """

    keywords: list[str] = []
    """
    List of keywords to **exact-match**.
    If any keywords are provided, at least 1 keyword must appear in the content,
    so use only if certain that the word will appear.
    Do not include keywords that can be covered in metadata_filters, e.g. dates, document type.
    If you are not certain that a word will appear, try using `text` for a semantic search instead.
    """

    metadata_filters: list[MetadataFilter] = Field(
        default=[],
        json_schema_extra={"metadata_schema": JournalDocumentMetadata.model_json_schema()},
    )
    """
    List of metadata conditions that must be matched.
    Refer to `metadata_schema` for the expected schema, as it exists in the database.
    """

    limit: int = Field(20, ge=3)
    """Maximum number of results to return."""
