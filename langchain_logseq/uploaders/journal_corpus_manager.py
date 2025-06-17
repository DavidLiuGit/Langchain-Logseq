from typing import Any

from pgvector_template.core import BaseCorpusManager


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
            "chunk_length": len(content),
            "word_count": len(content.split()),
        }
