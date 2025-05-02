from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_logseq.loaders import LogseqJournalLoader
from langchain_logseq.retrievers import LogseqJournalRetriever


class LogseqJournalDateRangeRetriever(LogseqJournalRetriever):
    """
    A retriever that retrieves documents from a Logseq journal within a specified date range.
    """
    
    def __init__(
        self,
        loader: LogseqJournalLoader,
    ):
        if not isinstance(loader, LogseqJournalLoader):
            raise TypeError("Loader must be an instance of LogseqJournalLoader")
        self.loader = loader


    def invoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> list[Document]:

