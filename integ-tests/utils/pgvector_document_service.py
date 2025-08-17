from logging import getLogger
from pytest import fixture

from langchain_aws import ChatBedrock
from pgvector_template.core import BaseDocumentOptionalProps
from pgvector_template.service import DocumentService, DocumentServiceConfig
from pgvector_template.db import TempDocumentDatabaseManager

from utils.api_bedrock import get_bedrock_client_from_environ
from utils.bedrock_embedder import BedrockEmbeddingProvider
from utils.logging import _enable_logging

from langchain_logseq.loaders import LogseqJournalFilesystemLoader, LogseqJournalLoaderInput
from langchain_logseq.retrievers.contextualizer import (
    RetrieverContextualizer,
    RetrieverContextualizerProps,
)
from langchain_logseq.retrievers.pgvector_journal_retriever import PGVectorJournalRetriever
from langchain_logseq.models.journal_pgvector import (
    JournalDocument,
    JournalCorpusMetadata,
    JournalDocumentMetadata,
    JournalSearchQuery,
)
from langchain_logseq.uploaders.pgvector.journal_corpus_manager import JournalCorpusManager


logger = getLogger(__name__)


@fixture
def pgvector_journal_retriever(pgvector_document_service):
    # integ tests require an LLM, which means we need to make API calls
    # set up the environment variables to make these calls. We will use AWS Bedrock.
    # use a low-cost Claude model for integ testing
    llm = ChatBedrock(
        client=get_bedrock_client_from_environ(),
        # model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        model_kwargs={
            "temperature": 0.5,
        },
    )

    # set up Retriever dependencies
    contextualizer = RetrieverContextualizer(
        RetrieverContextualizerProps(
            llm=llm,
            prompt=(
                "Given the user_input, and optional chat_history, create a search query object based "
                "on the schema provided, if you believe it is relevant. Do not include anything "
                "except for the schema, serialized as JSON. Do not answer the question directly"
            ),
            output_schema=JournalSearchQuery,
            enable_chat_history=True,
        )
    )
    yield PGVectorJournalRetriever(
        contextualizer=contextualizer,
        document_service=pgvector_document_service,
    )


@fixture
def pgvector_document_service(database_url: str):
    embedding_provider = BedrockEmbeddingProvider()

    # Setup database
    db_manager = TempDocumentDatabaseManager(
        database_url=database_url,
        schema_suffix="doc_service_e2e",
        document_classes=[JournalDocument],
    )
    temp_schema = db_manager.setup()

    # Set up DocumentService
    with db_manager.get_session() as session:
        doc_service_cfg = DocumentServiceConfig(
            document_cls=JournalDocument,
            embedding_provider=embedding_provider,
            corpus_manager_cls=JournalCorpusManager,
            document_metadata_cls=JournalDocumentMetadata,
        )
        document_service = DocumentService(session, doc_service_cfg)
        # at this stage, DocumentService is provisioned but has not data in it. Can yield here, or...

        # upload some documents so there's something to query against
        loader = LogseqJournalFilesystemLoader("./integ-tests/test_journals")
        loader_input = LogseqJournalLoaderInput(
            journal_start_date="2025-03-01",
            journal_end_date="2025-07-09",
            max_char_length=64 * 1024,
            enable_splitting=False,  # disable splitting; let JournalCorpusManager handle splitting
        )
        filesystem_docs: list = loader.load(loader_input)
        for fs_doc in filesystem_docs:
            corpus_md = JournalCorpusMetadata(date_str=fs_doc.metadata["journal_date"])
            optional_props = BaseDocumentOptionalProps(
                title=corpus_md.date_str,
                collection="integ-test-journals",
                original_url=f"./integ-tests/test_journals/{corpus_md.date_str.replace('-', '_')}.md",
                language="en",
            )
            logger.info(
                f"Uploading corpus with metadata={corpus_md}\n"
                f"\toptional_props={optional_props}\n"
                f"\tcontent preview: '{fs_doc.page_content[:32]}'"
            )
            document_service.corpus_manager.insert_corpus(
                fs_doc.page_content,
                corpus_md.model_dump(),
                optional_props,
                corpus_id=corpus_md.date_str,
            )

        yield document_service

    # cleanup
    db_manager.cleanup(temp_schema)
