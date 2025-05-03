import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.messages import HumanMessage, AIMessage
from langchain_logseq.retrievers.journal_date_range_retriever import LogseqJournalDateRangeRetriever
from langchain_logseq.retrievers.contextualizer import RetrieverContextualizer
from langchain_logseq.loaders import LogseqJournalLoader
from langchain_logseq.loaders.logseq_journal_loader_input import LogseqJournalLoaderInput



class TestLogseqJournalDateRangeRetriever(unittest.TestCase):
    def setUp(self):
        # Create a mock LogseqJournalLoaderInput for the contextualizer to return
        self.mock_loader_input = LogseqJournalLoaderInput(
            journal_start_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            journal_end_date=datetime.now().strftime("%Y-%m-%d"),
            max_char_length=8 * 1024
        )

        # Create a mock contextualizer that returns our mock loader input
        self.mock_contextualizer = Mock(spec=RetrieverContextualizer)
        self.mock_contextualizer._output_type = LogseqJournalLoaderInput
        self.mock_contextualizer.invoke = Mock(return_value=self.mock_loader_input)

        # Create a mock loader
        self.mock_loader = Mock(spec=LogseqJournalLoader)

        # Create the retriever with our mocks
        self.retriever = LogseqJournalDateRangeRetriever(
            self.mock_contextualizer,
            self.mock_loader
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
        self.assertEqual(result, self.mock_loader_input)


    def test_build_loader_input_with_chat_history(self):
        """Test _build_loader_input with query and chat history."""
        query = "What about meditation?"
        chat_history = [
            HumanMessage(content="What did I write about exercise last week?"),
            AIMessage(content="You wrote about running on Tuesday and yoga on Thursday.")
        ]
        
        # Call the method
        result = self.retriever._build_loader_input(query, chat_history)
        
        # Verify the contextualizer was called with the right parameters
        self.mock_contextualizer.invoke.assert_called_once()
        call_args = self.mock_contextualizer.invoke.call_args[0][0]
        self.assertEqual(call_args["user_input"], query)
        self.assertEqual(call_args["chat_history"], chat_history)
        
        # Verify the result is what we expect
        self.assertEqual(result, self.mock_loader_input)


    def test_build_loader_input_type_error(self):
        """Test _build_loader_input raises TypeError when contextualizer returns wrong type."""
        # Make the contextualizer return a string instead of LogseqJournalLoaderInput
        self.mock_contextualizer.invoke.return_value = "This is not a LogseqJournalLoaderInput"
        
        # Call the method and expect a TypeError
        with self.assertRaises(TypeError) as context:
            self.retriever._build_loader_input("What did I write about?")
        
        # Verify the error message
        self.assertIn("Expected LogseqJournalLoaderInput but got", str(context.exception))


    def test_get_relevant_documents(self):
        """Test _get_relevant_documents method."""
        # Setup
        query = "What did I write about last week?"
        mock_run_manager = Mock(spec=CallbackManagerForRetrieverRun)
        mock_documents = [MagicMock(), MagicMock()]
        self.mock_loader.load = Mock(return_value=mock_documents)
        
        # Execute
        result = self.retriever._get_relevant_documents(query, run_manager=mock_run_manager)
        
        # Verify
        self.mock_contextualizer.invoke.assert_called_once()
        self.mock_loader.load.assert_called_once_with(self.mock_loader_input)
        self.assertEqual(result, mock_documents)

    def test_get_relevant_documents_with_chat_history(self):
        """Test _get_relevant_documents with chat history."""
        # Setup
        query = "What about meditation?"
        chat_history = [
            HumanMessage(content="What did I write about exercise last week?"),
            AIMessage(content="You wrote about running on Tuesday and yoga on Thursday.")
        ]
        mock_run_manager = Mock(spec=CallbackManagerForRetrieverRun)
        mock_documents = [MagicMock(), MagicMock()]
        self.mock_loader.load = Mock(return_value=mock_documents)
        
        # Execute
        result = self.retriever._get_relevant_documents(
            query, 
            run_manager=mock_run_manager,
            chat_history=chat_history
        )
        
        # Verify
        self.mock_contextualizer.invoke.assert_called_once()
        call_args = self.mock_contextualizer.invoke.call_args[0][0]
        self.assertEqual(call_args["user_input"], query)
        self.assertEqual(call_args["chat_history"], chat_history)
        self.mock_loader.load.assert_called_once_with(self.mock_loader_input)
        self.assertEqual(result, mock_documents)


if __name__ == "__main__":
    unittest.main()
