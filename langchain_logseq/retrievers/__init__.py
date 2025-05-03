from abc import abstractmethod
from typing import TYPE_CHECKING

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig



class LogseqJournalRetriever(BaseRetriever):
    """
    A Langchain `Retriever` that is specifically for retrieving Logseq journal `Document`'s,
    based on a natural-language query. This `Retriever` will, in turn, leverage a Loader or
    Vectorstore to retrieve relevant documents to the query.
    """
    
    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """
        Called by `invoke`.
        
        `Retriever`s accept `input`, a natural-language "query", and returns a list of potentially
        relevant `Document`s to answer the query.
        This specific `Retriever` loads Logseq journal snippets from a date range, which will be determined
        internally.
        """
        raise NotImplementedError("This method shall be implemented by subclasses.")
