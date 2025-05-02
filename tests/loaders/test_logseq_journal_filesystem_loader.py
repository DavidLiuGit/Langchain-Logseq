import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_logseq.loaders.logseq_journal_filesystem_loader import LogseqJournalFilesystemLoader


class TestLogseqJournalFilesystemLoader(unittest.TestCase):
    def test_init_with_valid_path(self):
        """Test initialization with a valid path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test .md file
            test_file = Path(temp_dir) / "2025_03_27.md"
            test_file.write_text("Test content")
            
            # Initialize the loader
            loader = LogseqJournalFilesystemLoader(temp_dir)
            
            # Assert the path is set correctly
            self.assertEqual(loader.logseq_journal_path, temp_dir)


    def test_init_with_nonexistent_path(self):
        """Test initialization with a non-existent path."""
        non_existent_path = "/path/that/does/not/exist"
        
        # Assert that initialization raises ValueError
        with self.assertRaises(ValueError) as context:
            LogseqJournalFilesystemLoader(non_existent_path)
        
        self.assertIn(f"Logseq journal path does not exist: {non_existent_path}", str(context.exception))


    def test_init_with_file_path(self):
        """Test initialization with a path that is a file, not a directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            # Assert that initialization raises ValueError
            with self.assertRaises(ValueError) as context:
                LogseqJournalFilesystemLoader(temp_file.name)
            
            self.assertIn(f"Logseq journal path is not a directory: {temp_file.name}", str(context.exception))


    def test_empty_directory_warning(self):
        """Test that a warning is logged when the directory is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the logger to capture warnings
            with patch("langchain_logseq.loaders.logseq_journal_filesystem_loader.logger") as mock_logger:
                LogseqJournalFilesystemLoader(temp_dir)
                # Assert that both warnings were logged
                expected_calls = [
                    unittest.mock.call(f"Logseq journal directory is empty: {temp_dir}"),
                    unittest.mock.call(f"No files with .md extension found in {temp_dir}")
                ]
                mock_logger.warning.assert_has_calls(expected_calls)
                self.assertEqual(mock_logger.warning.call_count, 2)


    def test_no_md_files_warning(self):
        """Test that a warning is logged when the directory contains no .md files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a non-md file
            test_file = Path(temp_dir) / "not_a_journal.txt"
            test_file.write_text("Test content")
            
            # Mock the logger to capture warnings
            with patch("langchain_logseq.loaders.logseq_journal_filesystem_loader.logger") as mock_logger:
                LogseqJournalFilesystemLoader(temp_dir)
                
                # Assert that a warning was logged
                mock_logger.warning.assert_called_once_with(f"No files with .md extension found in {temp_dir}")


    def test_validate_logseq_journal_path(self):
        """Test the _validate_logseq_journal_path method directly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test .md file
            test_file = Path(temp_dir) / "2025_03_27.md"
            test_file.write_text("Test content")
            
            # Initialize the loader
            loader = LogseqJournalFilesystemLoader(temp_dir)
            
            # This should not raise any exceptions
            loader._validate_logseq_journal_path()



if __name__ == "__main__":
    unittest.main()
