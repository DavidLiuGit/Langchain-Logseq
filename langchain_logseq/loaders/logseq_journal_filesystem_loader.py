from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from logging import getLogger
from os import environ
from pathlib import Path

from langchain_logseq.loaders.logseq_journal_loader_input import LogseqJournalLoaderInput
import os


logger = getLogger(__name__)


class LogseqJournalFilesystemLoader(BaseLoader):
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
