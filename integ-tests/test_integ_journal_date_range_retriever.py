import unittest

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage

from langchain_logseq.loaders.journal_filesystem_loader import LogseqJournalFilesystemLoader
from langchain_logseq.loaders.journal_loader_input import LogseqJournalLoaderInput
from langchain_logseq.retrievers.contextualizer import RetrieverContextualizer, RetrieverContextualizerProps
from langchain_logseq.retrievers.journal_date_range_retriever import LogseqJournalDateRangeRetriever
from utils.api_bedrock import get_bedrock_client_from_environ
from utils.logging import _enable_logging


class TestIntegJournalDateRangeRetriever(unittest.TestCase):
    def setUp(self):
        _enable_logging()
        
        # integ tests require an LLM, which means we need to make API calls
        # set up the environment variables to make these calls. We will use AWS Bedrock.
        self.bedrock_client = get_bedrock_client_from_environ()

        # use a low-cost Claude model for integ testing
        self.llm = ChatBedrock(
            client=self.bedrock_client,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            model_kwargs={
                "temperature": 0.3,
            },
        )

        # set up Retriever dependencies
        loader = LogseqJournalFilesystemLoader("./integ-tests/test_journals")
        contextualizer = RetrieverContextualizer(
            RetrieverContextualizerProps(
                llm=self.llm,
                prompt=(
                    "Given the user_input, and optional chat_history, create an query object based"
                    "on the schema provided, if you believe it is relevant. Do not include anything"
                    "except for the schema, serialized as JSON. Do not answer the question directly"
                ),
                output_schema=LogseqJournalLoaderInput,
                enable_chat_history=True,
            )
        )
        self.retriever = LogseqJournalDateRangeRetriever(
            contextualizer,
            loader,
        )

    def test_retrieve_specific_date(self):
        """Use 2023-03-27 as an example."""
        query = "What did I do on Mar 27, 2025?"
        documents_retrieved = self.retriever.invoke({"input": query})
        print(documents_retrieved)
        self.assertIsNotNone(documents_retrieved)
        self.assertGreaterEqual(len(documents_retrieved), 3)  # should be at least 3, since 3 top-level bullets

        # mock a LLM response (since that's out of scope for this test), and retrieve more documents with the history
        history = [
            # intentionally obfuscate the date in the original query, to test history
            HumanMessage(content="What did I do on my dog's birthday?"),
            AIMessage(content="It looks like you had a lot of fun on 2025-03-27"),
        ]
        query = "What did I do on the next day?"
        documents_retrieved = self.retriever.invoke({"input": query, "chat_history": history})
        print(documents_retrieved)


if __name__ == "__main__":
    unittest.main()
