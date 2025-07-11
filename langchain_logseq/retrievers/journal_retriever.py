from abc import abstractmethod
from typing import Optional

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import BaseMessage


class LogseqJournalRetriever(BaseRetriever):
    """
    A Langchain `Retriever` that is specifically for retrieving Logseq journal `Document`'s,
    based on a natural-language query. This `Retriever` will, in turn, leverage a Loader or
    Vectorstore to retrieve relevant documents to the query.
    """

    def retrieve(self, query: str, chat_history: Optional[list[BaseMessage]] = None) -> list[Document]:
        """
        Called by `invoke` to retrieve relevant documents to the query.
        """
        return self._get_relevant_documents(query, chat_history=chat_history)

    @abstractmethod
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        chat_history: list[BaseMessage] | None = None,
    ) -> list[Document]:
        """
        Called by `invoke`.

        `Retriever`s accept `input`, a natural-language "query", and returns a list of potentially
        relevant `Document`s to answer the query.
        This specific `Retriever` loads Logseq journal snippets from a date range, which will be determined
        internally.
        """
        raise NotImplementedError("This method shall be implemented by subclasses.")
