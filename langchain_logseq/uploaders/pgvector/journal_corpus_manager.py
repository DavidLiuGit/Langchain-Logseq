import re
from typing import Any, Type

from pgvector_template.core import BaseCorpusManager, BaseCorpusManagerConfig, BaseDocument, BaseDocumentMetadata
from pydantic import Field

from langchain_logseq.models.journal_pgvector import JournalDocument, JournalDocumentMetadata


class JournalCorpusManagerConfig(BaseCorpusManagerConfig):
    """Configuration for Logseq journal `JournalCorpusManager`."""

    schema_name: str = "logseq_journal"
    """Name of the schema to use for the corpus manager"""
    document_cls: Type[BaseDocument] = JournalDocument
    """Class to use for the document model"""
    document_metadata_cls: Type[BaseDocumentMetadata] = JournalDocumentMetadata
    """Class to use for the document metadata model"""
    # embedding_provider: BaseEmbeddingProvider # is still required


class JournalCorpusManager(BaseCorpusManager):
    """
    CorpusManager declaration for Logseq journals. Each `Corpus` is the entire entry for a given date.
    """

    def _split_corpus(self, content: str, **kwargs) -> list[str]:
        """Split the journal file on root-level bullet points"""
        split_content = content.split("\n-")
        return [cleaned_chunk for chunk in split_content if (cleaned_chunk := chunk.strip().removeprefix("- "))]

    def _extract_chunk_metadata(self, content: str) -> dict[str, Any]:
        """Extract metadata from chunk content"""
        # Add some basic metadata about the chunk
        split_content = content.split()
        return {
            "chunk_len": len(content),
            "word_count": len(split_content),
            "references": self._extract_chunk_references(split_content),
            "anchor_ids": self._extract_anchor_ids(content),
        }

    def _extract_chunk_references(self, split_content: str) -> list[str]:
        """
        Extract references to other Logseq corpora, including other journals.
        Expected to start with `#`, e.g. `#2025-07-07`, `#cookout`.
        Special chars !?,:'"\ break references. \ is ignored.
        """
        references = []
        for word in split_content:
            if word.startswith("#"):
                ref = word.lstrip("#").rstrip("#").replace("\\", "")
                for char in "!?,:'\"":
                    ref = ref.split(char)[0]
                if ref:
                    references.append(ref)
        return references

    def _extract_anchor_ids(self, content: str) -> list[str]:
        """Extract Logseq anchor IDs from content (id:: <uuid>)"""

        return re.findall(r"id:: ([a-f0-9-]{36})", content)
