from typing import Any

from langchain_core.document_loaders import BaseLoader

# module exports
from langchain_logseq.loaders.journal_document_metadata import LogseqJournalDocumentMetadata
from langchain_logseq.loaders.journal_filesystem_loader import LogseqJournalLoader
from langchain_logseq.loaders.journal_loader_input import LogseqJournalLoaderInput


class LogseqJournalLoader(BaseLoader):
    """
    Base class for loading Logseq journal files.
    """

    def load(self, input: Any):
        raise NotImplementedError("This method should be implemented by subclasses.")
