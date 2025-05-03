from logging import getLogger
from typing import TYPE_CHECKING

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from langchain_logseq.loaders import LogseqJournalLoader
from langchain_logseq.retrievers import LogseqJournalRetriever
from langchain_logseq.retrievers.contextualizer import RetrieverContextualizer
from langchain_logseq.loaders.logseq_journal_loader_input import LogseqJournalLoaderInput


logger = getLogger(__name__)


class LogseqJournalDateRangeRetriever(LogseqJournalRetriever):
    """
    A retriever that retrieves documents from a Logseq journal within a specified date range.
    """
    
    def __init__(
        self,
        contextualizer: RetrieverContextualizer,
        loader: LogseqJournalLoader,
    ):
        if not isinstance(contextualizer, RetrieverContextualizer):
            raise TypeError("Contextualizer must be an instance of RetrieverContextualizer")
        self.contextualizer = contextualizer

        if not isinstance(loader, LogseqJournalLoader):
            raise TypeError("Loader must be an instance of LogseqJournalLoader")
        self.loader = loader


    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # determine the relevant 


    # def build_loader_input(self, query: str) -> LogseqJournalLoaderInput:
        """
        Based on the natural-language `query`, return an instance of `LogseqJournalLoaderInput`,
        which can then be used to invoke the `LogseqJournalLoader`.
        """
