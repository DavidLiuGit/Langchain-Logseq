from logging import getLogger
from typing import TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from pgvector_template.core import SearchQuery
from pgvector_template.service import DocumentService

from langchain_logseq.retrievers.contextualizer import RetrieverContextualizer
from langchain_logseq.retrievers.journal_retriever import LogseqJournalRetriever
from langchain_logseq.models.journal_pgvector import JournalDocument


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
        if not issubclass(contextualizer._output_type, SearchQuery):
            raise TypeError("contextualizer._output_type must be SearchQuery or a subclass")
        self._contextualizer = contextualizer

        if not isinstance(document_service, DocumentService):
            raise TypeError("document_service must be an instance of DocumentService")
        self._document_service = document_service
        self._verbose = verbose

    def _build_loader_input(
        self,
        query: str,
        chat_history: list[BaseMessage] = [],
    ) -> SearchQuery:
        """
        Based on the natural-language `query`, return an instance of `SearchQuery`,
        which can then be used to invoke the `DocumentService.search_client.search`.
        Use the `RetrieverContextualizer` to do this.
        """
        contextualizer_input = {
            "chat_history": chat_history,
            "user_input": query,
        }
        db_query = self._contextualizer.invoke(contextualizer_input)
        if self._verbose:
            logger.info(f"Contextualizer output: {db_query}")
        if not isinstance(db_query, SearchQuery):
            raise TypeError(f"Expected SearchQuery or subclass but got {type(db_query).__name__}")
        return db_query

    def _fetch_documents(self, loader_input: SearchQuery) -> list[Document]:
        """
        Return a list of `langchain_core.documents.Document`s based on the user's query
        (and chat_history if available).
        `load_input` shall be an instance of `SearchQuery` or a subclass, in this context.
        """
        db_results = self._document_service.search_client.search(loader_input)
        if self._verbose:
            logger.info(f"Retrieved {len(db_results)} documents from PGVector.")
        return [
            self._build_langchain_document_from_pgvector_document(result.document)
            for result in db_results
        ]

    def _build_langchain_document_from_pgvector_document(
        self, pgvector_document: JournalDocument
    ) -> Document:
        """
        Build a LangChain document from a PGVector document.
        """
        return Document(
            page_content=pgvector_document.content,
            metadata=pgvector_document.document_metadata,
        )
