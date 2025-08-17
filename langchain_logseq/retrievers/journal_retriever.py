from abc import abstractmethod
from logging import getLogger
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import BaseMessage
from pydantic import ValidationError


logger = getLogger(__name__)


class LogseqJournalRetriever(BaseRetriever):
    """
    A Langchain `Retriever` that is specifically for retrieving Logseq journal `Document`'s,
    based on a natural-language query. This `Retriever` will, in turn, leverage a Loader or
    Vectorstore to retrieve relevant documents to the query.
    """

    document_context: str = "These Documents represent journal entries. "

    def retrieve(self, query: str, chat_history: list[BaseMessage] | None = None) -> list[Document]:
        """
        Called by `invoke` to retrieve relevant documents to the query.
        """
        return self._get_relevant_documents(query, chat_history=chat_history)

    def _get_relevant_documents(
        self,
        query: str | dict[str, Any],
        *,
        run_manager: CallbackManagerForRetrieverRun,
        chat_history: list[BaseMessage] | None = None,
    ) -> list[Document]:
        """
        Called by `invoke`.

        `query` can be provided as a `str` (a natural-language query), or as a dict where
        `chat_history` can be provided additionally. Format:

        ```python
        query = {
            "user_input": "user's latest question",
            "chat_history": [("AiMessage", )]
        }
        ```

        Returns potentially relevant `langchain_core.documents.Document`s to answer the query.
        """
        # Handle case where query is passed as a dictionary (e.g., {"user_input": "query", "chat_history": [...]})
        if isinstance(query, dict):
            actual_query = query.get("user_input") or query.get("input") or query.get("query", "")
            chat_history = chat_history or query.get("chat_history") or query.get("history")
        else:
            actual_query = query

        try:
            loader_input = self._build_loader_input(actual_query, chat_history or [])
        except (TypeError, ValidationError, OutputParserException) as e:
            logger.exception("Error building loader input")
            return []

        return self._fetch_documents(loader_input)

    @abstractmethod
    def _fetch_documents(
        self,
        loader_input: Any,
    ) -> list[Document]:
        """
        Subclasses shall impl this method.
        Return a list of `langchain_core.documents.Document`s based on the user's query
        (and chat_history if available).
        """
        raise NotImplementedError("This method shall be implemented by subclasses.")

    @abstractmethod
    def _build_loader_input(
        self,
        query: str,
        chat_history: list[BaseMessage] = [],
    ) -> Any:
        """
        Subclasses shall impl this method.
        Return a dataclass, based on the user's query and chat_history if available, which shall
        be used in the subsequent step to load/query for relevant documents.
        """
        raise NotImplementedError("This method shall be implemented by subclasses.")
