from typing import Any

from langchain_logseq.retrievers import LogseqJournalRetriever


class LogseqJournalLoader(LogseqJournalRetriever):
    """
    Base class for loading Logseq journal files.
    """

    def load(self, input: Any):
        raise NotImplementedError("This method should be implemented by subclasses.")
