from logging import getLogger
from typing import Optional

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_logseq.loaders import LogseqJournalLoader
from langchain_core.messages import BaseMessage

from langchain_logseq.retrievers.journal_retriever import LogseqJournalRetriever
from langchain_logseq.retrievers.contextualizer import RetrieverContextualizer
from langchain_logseq.loaders.journal_loader_input import LogseqJournalLoaderInput


logger = getLogger(__name__)


class LogseqJournalDateRangeRetriever(LogseqJournalRetriever):
    """
    A `Retriever` that retrieves documents from a Logseq journal within a specified date range.
    """

    def __init__(
        self,
        contextualizer: RetrieverContextualizer,
        loader: LogseqJournalLoader,
        verbose: bool = True,
    ):
        """
        Initialize the `Retriever` with a contextualizer and a loader.

        Args:
            contextualizer (`RetrieverContextualizer`)
            loader (`LogseqJournalLoader`)
        """
        super().__init__()

        if not isinstance(contextualizer, RetrieverContextualizer):
            raise TypeError("Contextualizer must be an instance of RetrieverContextualizer")
        if contextualizer._output_type != LogseqJournalLoaderInput:
            raise TypeError("Contextualizer output type must be LogseqJournalLoaderInput")
        self._contextualizer = contextualizer

        if not isinstance(loader, LogseqJournalLoader):
            raise TypeError("Loader must be an instance of LogseqJournalLoader")
        self._loader = loader
        self._verbose = verbose

    # TODO: figure out how to provide chat_history when retriever used in a chain
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        chat_history: Optional[list[BaseMessage]] = None,
    ) -> list[Document]:
        loader_input = self._build_loader_input(query, chat_history or [])
        docs = self._loader.load(loader_input)
        if self._verbose:
            logger.info(f"Retrieved {len(docs)} documents")
        return docs

    def _build_loader_input(
        self,
        query: str,
        chat_history: list[BaseMessage] = [],
    ) -> LogseqJournalLoaderInput:
        """
        Based on the natural-language `query`, return an instance of `LogseqJournalLoaderInput`,
        which can then be used to invoke the `LogseqJournalLoader`.
        Use the `RetrieverContextualizer` to do this.
        """
        contextualizer_input = {
            "chat_history": chat_history,
            "user_input": query,
        }
        loader_input = self._contextualizer.invoke(contextualizer_input)
        if self._verbose:
            logger.info(f"Contextualizer output: {loader_input}")
        if not isinstance(loader_input, LogseqJournalLoaderInput):
            raise TypeError(f"Expected LogseqJournalLoaderInput but got {type(loader_input).__name__}")
        return loader_input
