import unittest
from unittest.mock import Mock, patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage
from pgvector_template.core import SearchQuery
from pgvector_template.service import DocumentService

from langchain_logseq.retrievers.pgvector_journal_retriever import PGVectorJournalRetriever
from langchain_logseq.retrievers.contextualizer import RetrieverContextualizer
from langchain_logseq.models.journal_pgvector import JournalDocument


class TestPGVectorJournalRetriever(unittest.TestCase):
    def setUp(self):
        # Create a mock SearchQuery for the contextualizer to return
        self.mock_search_query = Mock(spec=SearchQuery)

        # Create a mock contextualizer that returns our mock search query
        self.mock_contextualizer = Mock(spec=RetrieverContextualizer)
        self.mock_contextualizer._output_type = SearchQuery
        self.mock_contextualizer.invoke = Mock(return_value=self.mock_search_query)

        # Create mock search results
        self.mock_search_result1 = MagicMock()
        self.mock_search_result1.document = Mock(spec=JournalDocument)
        self.mock_search_result2 = MagicMock()
        self.mock_search_result2.document = Mock(spec=JournalDocument)
        self.mock_search_results = [self.mock_search_result1, self.mock_search_result2]

        # Create a mock document service
        self.mock_document_service = Mock(spec=DocumentService)
        self.mock_document_service.search_client = MagicMock()
        self.mock_document_service.search_client.search = Mock(
            return_value=self.mock_search_results
        )

        # Create the retriever with our mocks
        self.retriever = PGVectorJournalRetriever(
            contextualizer=self.mock_contextualizer, document_service=self.mock_document_service
        )

    def test_build_loader_input_with_query_only(self):
        """Test _build_loader_input with just a query string."""
        query = "What did I write about last week?"

        # Call the method
        result = self.retriever._build_loader_input(query)

        # Verify the contextualizer was called with the right parameters
        self.mock_contextualizer.invoke.assert_called_once()
        call_args = self.mock_contextualizer.invoke.call_args[0][0]
        self.assertEqual(call_args["user_input"], query)
        self.assertEqual(call_args["chat_history"], [])

        # Verify the result is what we expect
        self.assertEqual(result, self.mock_search_query)

    def test_build_loader_input_with_chat_history(self):
        """Test _build_loader_input with query and chat history."""
        query = "What about meditation?"
        chat_history = [
            HumanMessage(content="What did I write about exercise last week?"),
            AIMessage(content="You wrote about running on Tuesday and yoga on Thursday."),
        ]

        # Call the method
        result = self.retriever._build_loader_input(query, chat_history)

        # Verify the contextualizer was called with the right parameters
        self.mock_contextualizer.invoke.assert_called_once()
        call_args = self.mock_contextualizer.invoke.call_args[0][0]
        self.assertEqual(call_args["user_input"], query)
        self.assertEqual(call_args["chat_history"], chat_history)

        # Verify the result is what we expect
        self.assertEqual(result, self.mock_search_query)

    def test_build_loader_input_type_error(self):
        """Test _build_loader_input raises TypeError when contextualizer returns wrong type."""
        # Make the contextualizer return a string instead of SearchQuery
        self.mock_contextualizer.invoke.return_value = "This is not a SearchQuery"

        # Call the method and expect a TypeError
        with self.assertRaises(TypeError) as context:
            self.retriever._build_loader_input("What did I write about?")

        # Verify the error message
        self.assertIn("Expected SearchQuery but got", str(context.exception))

    def test_get_relevant_documents(self):
        """Test _get_relevant_documents method."""
        # Setup
        query = "What did I write about last week?"

        # Execute
        result = self.retriever._get_relevant_documents(query)

        # Verify
        self.mock_contextualizer.invoke.assert_called_once()
        self.mock_document_service.search_client.search.assert_called_once_with(
            self.mock_search_query
        )

        # Verify we get the documents from the search results
        expected_documents = [self.mock_search_result1.document, self.mock_search_result2.document]
        self.assertEqual(result, expected_documents)

    def test_get_relevant_documents_with_chat_history(self):
        """Test _get_relevant_documents with chat history."""
        # Setup
        query = "What about meditation?"
        chat_history = [
            HumanMessage(content="What did I write about exercise last week?"),
            AIMessage(content="You wrote about running on Tuesday and yoga on Thursday."),
        ]

        # Execute
        result = self.retriever._get_relevant_documents(query, chat_history=chat_history)

        # Verify
        self.mock_contextualizer.invoke.assert_called_once()
        call_args = self.mock_contextualizer.invoke.call_args[0][0]
        self.assertEqual(call_args["user_input"], query)
        self.assertEqual(call_args["chat_history"], chat_history)
        self.mock_document_service.search_client.search.assert_called_once_with(
            self.mock_search_query
        )

        # Verify we get the documents from the search results
        expected_documents = [self.mock_search_result1.document, self.mock_search_result2.document]
        self.assertEqual(result, expected_documents)

    def test_init_with_invalid_contextualizer(self):
        """Test initialization with invalid contextualizer."""
        # Create a contextualizer with wrong output type
        invalid_contextualizer = Mock(spec=RetrieverContextualizer)
        invalid_contextualizer._output_type = str  # Not JournalDocument

        # Expect TypeError when initializing with invalid contextualizer
        with self.assertRaises(TypeError) as context:
            PGVectorJournalRetriever(
                contextualizer=invalid_contextualizer, document_service=self.mock_document_service
            )

        self.assertIn("contextualizer._output_type must be", str(context.exception))

    def test_init_with_invalid_document_service(self):
        """Test initialization with invalid document service."""
        # Expect TypeError when initializing with invalid document service
        with self.assertRaises(TypeError) as context:
            PGVectorJournalRetriever(
                contextualizer=self.mock_contextualizer, document_service="not a DocumentService"
            )

        self.assertIn(
            "document_service must be an instance of DocumentService", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
