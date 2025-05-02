import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_logseq.loaders.logseq_journal_filesystem_loader import LogseqJournalFilesystemLoader
from langchain_logseq.loaders.logseq_journal_document_metadata import LogseqJournalDocumentMetadata


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


    def test_parse_journal_markdown_file_single_section(self):
        """Test parsing a markdown file with a single section."""
        filename = "2025_03_27.md"
        content = "This is a single section without bullet points"
        
        docs = LogseqJournalFilesystemLoader.parse_journal_markdown_file(content, filename)
        
        # Should create one document
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, content)
        self.assertEqual(docs[0].metadata["journal_date"], "2025-03-27")
        self.assertEqual(docs[0].metadata["journal_char_count"], len(content))


    def test_parse_journal_markdown_file_multiple_sections(self):
        """Test parsing a markdown file with multiple bullet point sections."""
        filename = "2025_03_27.md"
        content = "Header text\n- First bullet point\n- Second bullet point\n- Third bullet point"
        
        docs = LogseqJournalFilesystemLoader.parse_journal_markdown_file(content, filename)
        
        # Should create 4 documents (header + 3 bullet points)
        self.assertEqual(len(docs), 4)
        self.assertEqual(docs[0].page_content, "Header text")
        self.assertEqual(docs[1].page_content, "First bullet point")
        self.assertEqual(docs[2].page_content, "Second bullet point")
        self.assertEqual(docs[3].page_content, "Third bullet point")
        
        # All documents should have the same date metadata
        for doc in docs:
            self.assertEqual(doc.metadata["journal_date"], "2025-03-27")


    def test_parse_journal_markdown_file_metadata_correctness(self):
        """Test that the correct metadata is attached to each document."""
        filename = "2025_03_27.md"
        content = "Header\n- Bullet point"
        
        # Mock the parse_journal_markdown_file_metadata method to verify it's called correctly
        with patch.object(LogseqJournalFilesystemLoader, 'parse_journal_markdown_file_metadata') as mock_metadata:
            # Set up the mock to return a specific metadata object
            mock_metadata.return_value = LogseqJournalDocumentMetadata(
                journal_date="2025-03-27",
                journal_tags=[],
                journal_char_count=42
            )
            
            docs = LogseqJournalFilesystemLoader.parse_journal_markdown_file(content, filename)
            
            # Verify the method was called twice (once for each section)
            self.assertEqual(mock_metadata.call_count, 2)
            
            # Verify the metadata was attached to each document
            for doc in docs:
                self.assertEqual(doc.metadata["journal_date"], "2025-03-27")
                self.assertEqual(doc.metadata["journal_char_count"], 42)


    def test_parse_journal_markdown_file_metadata(self):
        """Test the parse_journal_markdown_file_metadata static method."""
        # Test with a simple filename and content
        filename = "2025_03_27.md"
        content = "This is some test content"
        
        metadata = LogseqJournalFilesystemLoader.parse_journal_markdown_file_metadata(content, filename)
        
        # Verify the metadata is correctly parsed
        self.assertEqual(metadata.journal_date, "2025-03-27")
        self.assertEqual(metadata.journal_tags, [])  # Currently hardcoded to empty list
        self.assertEqual(metadata.journal_char_count, len(content))


    def test_parse_journal_markdown_file_metadata_different_date_format(self):
        """Test the parse_journal_markdown_file_metadata with different date formats."""
        # Test with different date formats
        test_cases = [
            ("2025_03_27.md", "2025-03-27"),
            ("2023_01_01.md", "2023-01-01"),
            ("2024_12_31.md", "2024-12-31")
        ]
        
        for filename, expected_date in test_cases:
            content = f"Content for {filename}"
            metadata = LogseqJournalFilesystemLoader.parse_journal_markdown_file_metadata(content, filename)
            self.assertEqual(metadata.journal_date, expected_date)
            self.assertEqual(metadata.journal_char_count, len(content))


    def test_parse_journal_markdown_file_metadata_content_length(self):
        """Test that the character count is correctly calculated."""
        filename = "2025_03_27.md"
        
        # Test with different content lengths
        test_contents = [
            "",
            "Short content",
            "A" * 1000,  # 1000 characters
            "B" * 10000  # 10000 characters
        ]
        
        for content in test_contents:
            metadata = LogseqJournalFilesystemLoader.parse_journal_markdown_file_metadata(content, filename)
            self.assertEqual(metadata.journal_char_count, len(content))




if __name__ == "__main__":
    unittest.main()
