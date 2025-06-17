from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, String

from pgvector_template.core import BaseDocument


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


class 
