from abc import ABC, abstractmethod
from typing import Any

from logseq_retriever.models.document import Document


class LogseqJournalLoader(ABC):
    """
    Base class for loading Logseq journal files.
    """

    @abstractmethod
    def load(self, input: Any) -> list[Document]: ...
