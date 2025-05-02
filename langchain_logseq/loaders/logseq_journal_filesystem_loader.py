from logging import getLogger
from os import environ
from pathlib import Path

from langchain_core.documents import Document
from langchain_logseq.loaders import LogseqJournalLoader
from langchain_logseq.loaders.logseq_journal_loader_input import LogseqJournalLoaderInput
from langchain_logseq.loaders.logseq_journal_document_metadata import LogseqJournalDocumentMetadata
import os


logger = getLogger(__name__)


class LogseqJournalFilesystemLoader(LogseqJournalLoader):
    """
    Based on input, load a collection of Logseq journal files from the filesystem, as
    Langchain `Document`s.
    """
    
    def __init__(
        self,
        logseq_journal_path: str,
        **kwargs,
    ):
        """
        Initialize the loader with the path to the Logseq journal directory.
        `logseq_journal_path` should be contain Logesq journal files, such as `2025_03_27.md`
        """
        self.logseq_journal_path = logseq_journal_path
        self._validate_logseq_journal_path()


    def load(
        self,
        input: LogseqJournalLoaderInput,
    ) -> list[Document]:
        """
        Synchronously load the documents from the Logseq journal directory, according to the input.
        """
        if input.journal_end_date < input.journal_start_date:
            raise ValueError("journal_end_date must be after journal_start_date")
        
        # manipulate start_date & end_date to match the filename format
        start_date_fname = input.journal_start_date.replace("-", "_") + ".md"
        end_date_fname = input.journal_end_date.replace("-", "_") + ".md"

        documents: list[Document] = []
        # TODO this glob pattern can be improved by analyzing start_date & end_date to provide fewer matches
        for path in Path(self.logseq_journal_path).glob("*.md"):
            filename = path.name
            if self._match_journal(filename, start_date_fname, end_date_fname):
                file_path = os.path.join(self.logseq_journal_path, filename)
                with open(file_path, 'r') as file:
                    content = file.read()
                    documents.extend(self.__class__.parse_journal_markdown_file(content, filename))
        return documents


    def _validate_logseq_journal_path(self):
        """
        Validate the path to the Logseq journal directory. Check that the directory exists.
        If the directory is empty, or does not contain files with the expected format, log a warning.
        """
        # verify that the path exist, and is a directory
        if not os.path.exists(self.logseq_journal_path):
            raise ValueError(f"Logseq journal path does not exist: {self.logseq_journal_path}")
        if not os.path.isdir(self.logseq_journal_path):
            raise ValueError(f"Logseq journal path is not a directory: {self.logseq_journal_path}")
        
        # verify that the directory contains files with the expected format
        files = os.listdir(self.logseq_journal_path)
        if len(files) == 0:
            logger.warning(f"Logseq journal directory is empty: {self.logseq_journal_path}")
        files = Path(self.logseq_journal_path).glob("*.md")
        if not len(list(files)) > 0:
            logger.warning(f"No files with .md extension found in {self.logseq_journal_path}")


    def _match_journal(self, filename: str, start_date_fname: str, end_date_fname: str) -> bool:
        """Return `True` if journal date is between `journal_start_date` & `journal_end_date`"""
        return start_date_fname <= filename and filename <= end_date_fname
    
    
    @staticmethod
    def parse_journal_markdown_file(content: str, filename: str) -> list[Document]:
        """
        Generate `Document`s from a file's contents. If necessary, split content into digestible
        `Document`s, and attach metadata.
        This function can potentially be augmented by calling Logseq APIs, rather than simply parsing markdown files.
        """
        sections = content.split('\n- ')
        docs = []
        for section in sections:
            if section_content := section.strip():
                # Create a Document
                # first, check that the content length (char count) is acceptable
                # if longer than acceptable, then call recursively
                # TODO: use self.p.max_char_count below instead
                metadata = LogseqJournalFilesystemLoader.parse_journal_markdown_file_metadata(
                    section_content, filename)
                docs.append(Document(page_content=section_content, metadata=metadata.model_dump()))
        return docs


    @staticmethod
    def parse_journal_markdown_file_metadata(section: str, filename: str) -> LogseqJournalDocumentMetadata:
        """
        Parse metadata from a journal markdown file. Return `LogseqMarkdownDocumentMetadata`.
        This function can potentially be augmented by calling Logseq APIs, rather than simply parsing markdown files.
        """
        # Extract date from filename
        date_str = filename.replace('.md', '').replace('_', '-')
        char_count = len(section)

        return LogseqJournalDocumentMetadata(
            journal_date=date_str,
            # TODO get tags from Document's contents
            journal_tags=[],
            journal_char_count=char_count,
        )
        