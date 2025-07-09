from pydantic import Field
from typing import Any, Type

from pgvector_template.core import BaseCorpusManager, BaseCorpusManagerConfig, BaseDocument, BaseDocumentMetadata

from langchain_logseq.models.journal_pgvector_template import JournalDocument, JournalDocumentMetadata


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
        # initial version: split on root-level bullet points only, i.e. `\n-`
        split_content = content.split('\n-')
        return [chunk for chunk in split_content if len(chunk.strip())]

    def _extract_chunk_metadata(self, content: str) -> dict[str, Any]:
        """Extract metadata from chunk content"""
        # Add some basic metadata about the chunk
        return {
            "chunk_len": len(content),
            "word_count": len(content.split()),
        }
