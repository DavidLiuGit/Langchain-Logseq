from logging import getLogger
from sqlalchemy.orm import Session

from langchain_core.messages import BaseMessage
from pgvector_template.core import SearchQuery
from pgvector_template.service import DocumentService

from langchain_logseq.retrievers.contextualizer import RetrieverContextualizer
from langchain_logseq.retrievers.journal_retriever import LogseqJournalRetriever
from langchain_logseq.models.journal_pgvector import (
    JournalDocument,
    JournalCorpusMetadata,
    JournalDocumentMetadata,
    JournalSearchClientConfig,
)


logger = getLogger(__name__)


class PGVectorJournalRetriever(LogseqJournalRetriever):
    """
    A `Retriever` that relies on a PGVector backend to fetch Logseq journals.
    """

    def __init__(
        self,
        contextualizer: RetrieverContextualizer,
        document_service: DocumentService,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize the `Retriever` with a contextualizer and a loader.
        """
        super().__init__()

        if not isinstance(contextualizer, RetrieverContextualizer):
            raise TypeError("contextualizer must be an instance of RetrieverContextualizer")
        if contextualizer._output_type != SearchQuery:
            raise TypeError("contextualizer._output_type must be SearchQuery")
        self._contextualizer = contextualizer

        if not isinstance(document_service, DocumentService):
            raise TypeError("document_service must be an instance of DocumentService")
        self._document_service = document_service
        self._verbose = verbose

    def _get_relevant_documents(
        self,
        query: str,
        *,
        # run_manager: CallbackManagerForRetrieverRun,
        chat_history: list[BaseMessage] | None = None,
    ) -> list[JournalDocument]:
        db_query = self._build_loader_input(query, chat_history or [])
        db_results = self._document_service.search_client.search(db_query)
        if self._verbose:
            logger.info(f"Retrieved {len(db_results)} documents from PGVector.")
        return [result.document for result in db_results]

    def _build_loader_input(
        self,
        user_query: str,
        chat_history: list[BaseMessage] = [],
    ) -> SearchQuery:
        """
        Based on the natural-language `query`, return an instance of `SearchQuery`,
        which can then be used to invoke the `DocumentService.search_client.search`.
        Use the `RetrieverContextualizer` to do this.
        """
        contextualizer_input = {
            "chat_history": chat_history,
            "user_input": user_query,
        }
        db_query = self._contextualizer.invoke(contextualizer_input)
        if self._verbose:
            logger.info(f"Contextualizer output: {db_query}")
        if not isinstance(db_query, SearchQuery):
            raise TypeError(f"Expected SearchQuery but got {type(db_query).__name__}")
        return db_query
