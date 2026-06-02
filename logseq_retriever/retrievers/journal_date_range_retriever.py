from collections.abc import Sequence
from logging import getLogger

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from logseq_retriever.loaders import LogseqJournalLoader
from logseq_retriever.retrievers.journal_retriever import LogseqJournalRetriever
from logseq_retriever.retrievers.contextualizer import RetrieverContextualizer
from logseq_retriever.loaders.journal_loader_input import LogseqJournalLoaderInput


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
            raise TypeError(
                "Contextualizer must be an instance of RetrieverContextualizer"
            )
        if contextualizer._output_type != LogseqJournalLoaderInput:
            raise TypeError(
                "Contextualizer output type must be LogseqJournalLoaderInput"
            )
        self._contextualizer = contextualizer

        if not isinstance(loader, LogseqJournalLoader):
            raise TypeError("Loader must be an instance of LogseqJournalLoader")
        self._loader = loader
        self._verbose = verbose

    def _build_loader_input(
        self,
        query: str,
        chat_history: Sequence[BaseMessage] = (),
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
            raise TypeError(
                f"Expected LogseqJournalLoaderInput but got {type(loader_input).__name__}"
            )
        return loader_input

    def _fetch_documents(
        self, loader_input: LogseqJournalLoaderInput
    ) -> list[Document]:
        docs = self._loader.load(loader_input)
        if self._verbose:
            logger.info(f"Retrieved {len(docs)} documents")
        return docs
