from typing import Type

from pgvector.sqlalchemy import Vector
from pydantic import Field
from sqlalchemy import Column, String

from pgvector_template.core import (
    BaseDocument,
    BaseDocumentMetadata,
    BaseSearchClient,
    BaseSearchClientConfig,
    SearchQuery,
)
from sqlalchemy.orm import Session


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
    date_str: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    # defaults
    document_type: str = Field(default="logseq_journal")
    schema_version: str = Field(default="2025-07-10")


class JournalDocumentMetadata(JournalCorpusMetadata):
    """Metadata schema for Logseq journal `Document`s. 1-or-more `Document`s make up a corpus."""

    # chunk/document
    """Date in ISO format, e.g. `2025-04-20`"""
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

    document_cls: Type[JournalDocument] = JournalDocument
    """The document type to use for the search client."""
    document_metadata_cls: Type[JournalDocumentMetadata] = JournalDocumentMetadata
    """The document metadata type to use for the search client."""
    # embedding_provider
