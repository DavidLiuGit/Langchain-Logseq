from typing import Any

from langchain_core.document_loaders import BaseLoader


class LogseqJournalLoader(BaseLoader):
    """
    Base class for loading Logseq journal files.
    """

    def load(self, input: Any):
        raise NotImplementedError("This method should be implemented by subclasses.")
