import argparse
import os
import sys
from datetime import datetime
from logging import getLogger
from pathlib import Path

from langchain_core.documents import Document as LangchainDocument
from pgvector_template.db import TempDocumentDatabaseManager
from pgvector_template.core import BaseDocumentOptionalProps

from langchain_logseq.models.journal_pgvector import JournalDocument, JournalCorpusMetadata
from langchain_logseq.loaders import LogseqJournalFilesystemLoader, LogseqJournalLoaderInput
from langchain_logseq.uploaders.pgvector import JournalCorpusManager, JournalCorpusManagerConfig
from pgvector_utils.db_util import database_url
from utils.bedrock_embedder import BedrockEmbeddingProvider
from utils.logging import setup_logging
from dotenv import load_dotenv

load_dotenv()
setup_logging()
logger = getLogger(__name__)

################################################################################
##### CLIENTS
################################################################################


################################################################################
##### HELPERS
################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="Upload Logseq journals to pgvector")

    parser.add_argument(
        "-p",
        "--path",
        default=os.getenv("LOGSEQ_JOURNAL_PATH"),
        help="Location of Logseq journal directory (default: LOGSEQ_JOURNAL_PATH env var)",
    )
    parser.add_argument("from_date", help="Start date (inclusive), format: YYYY-MM-DD")
    parser.add_argument("to_date", help="End date (inclusive), format: YYYY-MM-DD")

    args = parser.parse_args()

    # Validate path
    if not args.path:
        raise ValueError("Path must be provided via -p/--path or LOGSEQ_JOURNAL_PATH env var")

    path = Path(args.path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid path: {args.path} is not a valid directory")

    # Validate dates
    try:
        datetime.strptime(args.from_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid from_date format: {args.from_date}. Expected YYYY-MM-DD")

    try:
        datetime.strptime(args.to_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid to_date format: {args.to_date}. Expected YYYY-MM-DD")

    return args


def setup_journal_filesystem_loader(args) -> LogseqJournalFilesystemLoader:
    """
    Setup the LogseqJournalFilesystemLoader with the provided path and date range.
    """
    loader = LogseqJournalFilesystemLoader(
        logseq_journal_path=args.path,
        from_date=args.from_date,
        to_date=args.to_date,
    )
    return loader


def build_db_optional_props(args, collection: str, corpus_md: JournalCorpusMetadata) -> BaseDocumentOptionalProps:
    # 1 table can host multiple collections of the same type
    return BaseDocumentOptionalProps(
        title=corpus_md.date_str,
        collection=collection,
        original_url=f"{args.path}/{corpus_md.date_str.replace('-', '_')}.md",
        language="en",
    )


################################################################################
##### MAIN
################################################################################


def main():
    args = parse_args()
    db_url = database_url()

    # set up clients for: Loader, DB, embedder
    loader = setup_journal_filesystem_loader(args)
    db_manager = TempDocumentDatabaseManager(db_url, "logseq", [JournalDocument])
    temp_schema_name = db_manager.setup()
    embedder = BedrockEmbeddingProvider()

    with db_manager.get_session() as session:
        corpus_manager = JournalCorpusManager(
            session,
            JournalCorpusManagerConfig(schema_name=temp_schema_name, embedding_provider=embedder),
        )

        # load documents using loader
        loader_input = LogseqJournalLoaderInput(
            journal_start_date=args.from_date,
            journal_end_date=args.to_date,
            max_char_length=64 * 1024,
            enable_splitting=False,  # disable splitting; let JournalCorpusManager handle splitting
        )
        filesystem_docs: list[LangchainDocument] = loader.load(loader_input)

        # process documents from the filesystem for upload

        for fs_doc in filesystem_docs:
            corpus_md = JournalCorpusMetadata(date_str=fs_doc.metadata["journal_date"])
            optional_props = build_db_optional_props(args, "Foo's Journal Collection", corpus_md)
            logger.info(
                f"Uploading corpus with metadata={corpus_md}\n"
                f"\toptional_props={optional_props}\n"
                f"\tcontent preview: '{fs_doc.page_content[:32]}'"
            )
            corpus_manager.insert_corpus(
                fs_doc.page_content, corpus_md.model_dump(), optional_props, corpus_id=corpus_md.date_str
            )


if __name__ == "__main__":
    main()
