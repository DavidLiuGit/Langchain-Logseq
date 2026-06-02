from typing import Any

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class LogseqJournalLoader(BaseLoader):
    """
    Base class for loading Logseq journal files.
    """

    def load(self, input: Any) -> list[Document]:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        raise NotImplementedError("This method should be implemented by subclasses.")
