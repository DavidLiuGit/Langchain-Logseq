from pydantic import BaseModel, Field
from textwrap import dedent
from typing import List, Optional
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_logseq.retrievers.contextualizer import RetrieverContextualizer, RetrieverContextualizerProps


class TestRetrieverContextualizer(unittest.TestCase):
    def setUp(self):
        # Create a mock LLM for testing
        self.mock_llm = MagicMock(spec=BaseLanguageModel)
        self.mock_llm.invoke.return_value = "Standalone question: What is the capital of France?"
        
        # Define a basic prompt for testing
        self.test_prompt = dedent("""
            Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n
            Chat History:
            {chat_history}\n
            Follow Up Input: {user_input}\n
            Standalone question:
            """)


    def test_init_with_valid_props(self):
        """Test initialization with valid properties."""
        props = RetrieverContextualizerProps(
            llm=self.mock_llm,
            prompt=self.test_prompt,
            output_schema=None
        )

        contextualizer = RetrieverContextualizer(props)

        self.assertEqual(contextualizer.props, props)
        self.assertIsNotNone(contextualizer.chain)


    def test_generate_chain_without_output_schema(self):
        """Test chain generation without an output schema."""
        props = RetrieverContextualizerProps(
            llm=self.mock_llm,
            prompt=self.test_prompt,
            output_schema=None,
        )
        
        contextualizer = RetrieverContextualizer(props)
        
        # Check that the chain components are correctly set up
        self.assertIsInstance(contextualizer.chain, type(PromptTemplate.from_template(self.test_prompt) | self.mock_llm | StrOutputParser()))


    def test_generate_chain_with_output_schema(self):
        """Test chain generation with an output schema."""
        # Define a test output schema
        question_description = "The standalone question"
        class TestOutputSchema(BaseModel):
            question: str = Field(description=question_description)
            
        props = RetrieverContextualizerProps(
            llm=self.mock_llm,
            prompt=self.test_prompt,
            output_schema=TestOutputSchema
        )
        
        contextualizer = RetrieverContextualizer(props)
        
        # Check that the parser is correctly set up
        self.assertIsInstance(contextualizer.parser, PydanticOutputParser)
        self.assertEqual(contextualizer.parser.pydantic_object, TestOutputSchema)


    @patch('langchain_core.prompts.PromptTemplate.from_template')
    def test_invoke_without_output_schema(self, mock_from_template):
        """Test invoking the contextualizer without an output schema."""
        # Setup the mock chain
        mock_chain = MagicMock()
        mock_from_template.return_value = mock_chain
        mock_chain.__or__ = MagicMock(return_value=mock_chain)
        
        props = RetrieverContextualizerProps(
            llm=self.mock_llm,
            prompt=self.test_prompt
        )
        
        contextualizer = RetrieverContextualizer(props)
        contextualizer.chain = mock_chain
        
        # Test invoking with input
        test_input = {"user_input": "What is Paris?", "chat_history": "We were discussing France."}
        contextualizer.invoke(test_input)
        
        # Check that the chain was called with the correct input
        mock_chain.invoke.assert_called_once_with(test_input, config=None)


    @patch('langchain_core.prompts.PromptTemplate.from_template')
    def test_invoke_with_output_schema(self, mock_from_template):
        """Test invoking the contextualizer with an output schema."""
        # Define a test output schema
        class TestOutputSchema(BaseModel):
            question: str = Field(description="The standalone question")
        
        # Setup the mock chain
        mock_chain = MagicMock()
        mock_from_template.return_value = mock_chain
        mock_chain.__or__ = MagicMock(return_value=mock_chain)
        
        props = RetrieverContextualizerProps(
            llm=self.mock_llm,
            prompt=self.test_prompt,
            output_schema=TestOutputSchema
        )
        
        contextualizer = RetrieverContextualizer(props)
        contextualizer.chain = mock_chain
        
        # Test invoking with input
        test_input = {"user_input": "What is Paris?", "chat_history": "We were discussing France."}
        contextualizer.invoke(test_input)
        
        # Check that the chain was called with the correct input
        mock_chain.invoke.assert_called_once_with(test_input, config=None)


    def test_integration_without_output_schema(self):
        """Integration test for the contextualizer without an output schema."""
        props = RetrieverContextualizerProps(
            llm=self.mock_llm,
            prompt=self.test_prompt
        )
        
        contextualizer = RetrieverContextualizer(props)
        
        # Test invoking with input
        test_input = {"user_input": "What is Paris?", "chat_history": "We were discussing France."}
        result = contextualizer.invoke(test_input)
        
        # The mock LLM should return our predefined response
        self.assertEqual(result, "Standalone question: What is the capital of France?")


    def test_integration_with_output_schema(self):
        """Integration test for the contextualizer with an output schema."""
        # Define a test output schema
        class TestOutputSchema(BaseModel):
            question: str = Field(description="The standalone question")
        
        # Configure the mock LLM to return a valid JSON string for the parser
        self.mock_llm.invoke.return_value = '{"question": "What is the capital of France?"}'
        
        props = RetrieverContextualizerProps(
            llm=self.mock_llm,
            prompt=self.test_prompt,
            output_schema=TestOutputSchema
        )
        
        contextualizer = RetrieverContextualizer(props)
        from pprint import pprint
        # pprint(contextualizer.parser.get_format_instructions())
        # pprint(contextualizer.prompt_template.template)
        
        # Test invoking with input
        test_input = {
            "user_input": "What is Paris?",
            "chat_history": "We were discussing France.",
        }
        result = contextualizer.invoke(test_input)
        
        # Check that the result is a TestOutputSchema instance with the expected value
        self.assertIsInstance(result, TestOutputSchema)
        self.assertEqual(result.question, "What is the capital of France?")


if __name__ == "__main__":
    unittest.main()