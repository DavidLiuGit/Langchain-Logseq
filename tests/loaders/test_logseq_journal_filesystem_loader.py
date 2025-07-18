import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from langchain_logseq.loaders.journal_filesystem_loader import LogseqJournalFilesystemLoader
from langchain_logseq.loaders.journal_document_metadata import LogseqJournalDocumentMetadata
from langchain_logseq.loaders.journal_loader_input import LogseqJournalLoaderInput


class TestLogseqJournalFilesystemLoader(unittest.TestCase):
    ###########################################################################
    ##### Constructor & validator tests
    ###########################################################################
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
            with patch("langchain_logseq.loaders.journal_filesystem_loader.logger") as mock_logger:
                LogseqJournalFilesystemLoader(temp_dir)
                # Assert that both warnings were logged
                expected_calls = [
                    unittest.mock.call(f"Logseq journal directory is empty: {temp_dir}"),
                    unittest.mock.call(f"No files with .md extension found in {temp_dir}"),
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
            with patch("langchain_logseq.loaders.journal_filesystem_loader.logger") as mock_logger:
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

    ###########################################################################
    ##### load() tests
    ###########################################################################
    def test_load_invalid_date_range(self):
        """Test that load raises ValueError when end date is before start date."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = Path(temp_dir) / "2025_03_27.md"
            test_file.write_text("Test content")

            loader = LogseqJournalFilesystemLoader(temp_dir)

            # Create input with invalid date range (end before start)
            input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-27", journal_end_date="2025-03-26")

            # Assert that load raises ValueError
            with self.assertRaises(ValueError) as context:
                loader.load(input_data)

            self.assertIn("journal_end_date must be after journal_start_date", str(context.exception))

    def test_load_no_matching_files(self):
        """Test that load returns empty list when no files match the date range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files outside the date range
            test_file1 = Path(temp_dir) / "2025_03_27.md"
            test_file1.write_text("Test content 1")
            test_file2 = Path(temp_dir) / "2025_04_15.md"
            test_file2.write_text("Test content 2")

            loader = LogseqJournalFilesystemLoader(temp_dir)

            # Create input with date range that doesn't match any files
            input_data = LogseqJournalLoaderInput(journal_start_date="2025-05-01", journal_end_date="2025-05-31")

            # Load should return empty list
            documents = loader.load(input_data)
            self.assertEqual(len(documents), 0)

    def test_load_single_matching_file(self):
        """Test loading a single file that matches the date range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file1 = Path(temp_dir) / "2025_03_27.md"
            test_file1.write_text("Test content")
            test_file2 = Path(temp_dir) / "2025_05_15.md"
            test_file2.write_text("Outside range")

            loader = LogseqJournalFilesystemLoader(temp_dir)

            # Create input with date range that matches one file
            input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-01", journal_end_date="2025-04-01")

            # Mock parse_journal_markdown_file to isolate the test
            expected_docs = [Document(page_content="Mocked content", metadata={"mocked": True})]
            with patch.object(
                LogseqJournalFilesystemLoader, "parse_journal_markdown_file", return_value=expected_docs
            ) as mock_parse:
                documents = loader.load(input_data)

                # Verify parse_journal_markdown_file was called once with the right parameters
                mock_parse.assert_called_once()
                args, _ = mock_parse.call_args
                self.assertEqual(args[1], "2025_03_27.md")  # filename

                # Verify the returned documents
                self.assertEqual(documents, expected_docs)

    def test_load_multiple_matching_files(self):
        """Test loading multiple files that match the date range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file1 = Path(temp_dir) / "2025_03_27.md"
            test_file1.write_text("March content")
            test_file2 = Path(temp_dir) / "2025_04_15.md"
            test_file2.write_text("April content")
            test_file3 = Path(temp_dir) / "2025_05_10.md"
            test_file3.write_text("May content")

            loader = LogseqJournalFilesystemLoader(temp_dir)

            # Create input with date range that matches two files
            input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-15", journal_end_date="2025-04-30")

            # Mock parse_journal_markdown_file to return different documents for each file
            def mock_parse_side_effect(content, filename, enable_splitting=True):
                if filename == "2025_03_27.md":
                    return [Document(page_content="March doc", metadata={"file": "march"})]
                elif filename == "2025_04_15.md":
                    return [Document(page_content="April doc", metadata={"file": "april"})]
                return []

            with patch.object(
                LogseqJournalFilesystemLoader, "parse_journal_markdown_file", side_effect=mock_parse_side_effect
            ) as mock_parse:
                documents = loader.load(input_data)

                # Verify parse_journal_markdown_file was called twice
                self.assertEqual(mock_parse.call_count, 2)

                # Verify the returned documents (2 documents, one from each file)
                self.assertEqual(len(documents), 2)
                self.assertEqual(documents[0].page_content, "March doc")
                self.assertEqual(documents[1].page_content, "April doc")

    def test_load_exact_date_range(self):
        """Test loading with exact start and end dates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_file1 = Path(temp_dir) / "2025_03_27.md"
            test_file1.write_text("March 27 content")
            test_file2 = Path(temp_dir) / "2025_03_28.md"
            test_file2.write_text("March 28 content")
            test_file3 = Path(temp_dir) / "2025_03_29.md"
            test_file3.write_text("March 29 content")

            loader = LogseqJournalFilesystemLoader(temp_dir)

            # Create input with exact date range
            input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-27", journal_end_date="2025-03-29")

            # Mock parse_journal_markdown_file to return a single document for each file
            def mock_parse_side_effect(content, filename, enable_splitting=True):
                return [Document(page_content=f"Content from {filename}", metadata={"filename": filename})]

            with patch.object(
                LogseqJournalFilesystemLoader, "parse_journal_markdown_file", side_effect=mock_parse_side_effect
            ) as mock_parse:
                documents = loader.load(input_data)

                # Verify parse_journal_markdown_file was called for all three files
                self.assertEqual(mock_parse.call_count, 3)

                # Verify the returned documents
                self.assertEqual(len(documents), 3)
                filenames = [doc.metadata["filename"] for doc in documents]
                self.assertIn("2025_03_27.md", filenames)
                self.assertIn("2025_03_28.md", filenames)
                self.assertIn("2025_03_29.md", filenames)

    def test_load_integration_with_parse_methods(self):
        """Integration test for load method with actual parse methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file with multiple sections
            test_file = Path(temp_dir) / "2025_03_27.md"
            test_file.write_text("Header\n- First bullet\n- Second bullet")

            loader = LogseqJournalFilesystemLoader(temp_dir)

            # Create input that matches the test file
            input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-27", journal_end_date="2025-03-27")

            # Call load without mocking the parse methods
            documents = loader.load(input_data)

            # Verify the documents were created correctly
            self.assertEqual(len(documents), 3)  # Header + 2 bullets
            self.assertEqual(documents[0].page_content, "Header")
            self.assertEqual(documents[1].page_content, "First bullet")
            self.assertEqual(documents[2].page_content, "Second bullet")

            # Verify metadata
            for doc in documents:
                self.assertEqual(doc.metadata["journal_date"], "2025-03-27")

    ###########################################################################
    ##### parse_journal_markdown_file() tests
    ###########################################################################
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

    def test_parse_journal_markdown_file_no_splitting(self):
        """Test parsing a markdown file with enable_splitting=False."""
        filename = "2025_03_27.md"
        content = "Header text\n- First bullet point\n- Second bullet point"

        docs = LogseqJournalFilesystemLoader.parse_journal_markdown_file(content, filename, enable_splitting=False)

        # Should create only one document when splitting is disabled
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, content)
        self.assertEqual(docs[0].metadata["journal_date"], "2025-03-27")
        self.assertEqual(docs[0].metadata["journal_char_count"], len(content))

    ###########################################################################
    ##### parse_journal_markdown_file_metadata() tests
    ###########################################################################
    def test_parse_journal_markdown_file_metadata_correctness(self):
        """Test that the correct metadata is attached to each document."""
        filename = "2025_03_27.md"
        content = "Header\n- Bullet point"

        # Mock the parse_journal_markdown_file_metadata method to verify it's called correctly
        with patch.object(LogseqJournalFilesystemLoader, "parse_journal_markdown_file_metadata") as mock_metadata:
            # Set up the mock to return a specific metadata object
            mock_metadata.return_value = LogseqJournalDocumentMetadata(
                journal_date="2025-03-27", journal_tags=[], journal_char_count=42
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
        test_cases = [("2025_03_27.md", "2025-03-27"), ("2023_01_01.md", "2023-01-01"), ("2024_12_31.md", "2024-12-31")]

        for filename, expected_date in test_cases:
            content = f"Content for {filename}"
            metadata = LogseqJournalFilesystemLoader.parse_journal_markdown_file_metadata(content, filename)
            self.assertEqual(metadata.journal_date, expected_date)
            self.assertEqual(metadata.journal_char_count, len(content))

    def test_parse_journal_markdown_file_metadata_content_length(self):
        """Test that the character count is correctly calculated."""
        filename = "2025_03_27.md"

        # Test with different content lengths
        test_contents = ["", "Short content", "A" * 1000, "B" * 10000]  # 1000 characters  # 10000 characters

        for content in test_contents:
            metadata = LogseqJournalFilesystemLoader.parse_journal_markdown_file_metadata(content, filename)
            self.assertEqual(metadata.journal_char_count, len(content))


class TestLogseqJournalFilesystemLoaderWithRealFiles(unittest.TestCase):
    """Tests for LogseqJournalFilesystemLoader using real Logseq journal files."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        # Path to the test journals directory
        cls.journals_dir = Path(__file__).parent / "test_journals"

        # Ensure the test journals directory exists
        if not cls.journals_dir.exists():
            raise ValueError(f"Test journals directory does not exist: {cls.journals_dir}")

        # Initialize the loader
        cls.loader = LogseqJournalFilesystemLoader(str(cls.journals_dir))

    def test_load_single_day(self):
        """Test loading a single day's journal."""
        input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-27", journal_end_date="2025-03-27")

        documents = self.loader.load(input_data)

        # Check that we got the expected number of documents
        self.assertGreater(len(documents), 0)

        # Check that all documents have the correct date
        for doc in documents:
            self.assertEqual(doc.metadata["journal_date"], "2025-03-27")

        # Check that content from the file is present
        content_text = " ".join([doc.page_content for doc in documents])
        self.assertIn("Morning thoughts", content_text)
        self.assertIn("Afternoon update", content_text)
        self.assertIn("Evening reflection", content_text)

    def test_load_date_range(self):
        """Test loading journals across a date range."""
        input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-27", journal_end_date="2025-03-29")

        documents = self.loader.load(input_data)

        # Check that we got documents from all three files
        dates = set(doc.metadata["journal_date"] for doc in documents)
        self.assertEqual(len(dates), 3)
        self.assertIn("2025-03-27", dates)
        self.assertIn("2025-03-28", dates)
        self.assertIn("2025-03-29", dates)

        # Check for content from different files
        content_text = " ".join([doc.page_content for doc in documents])
        self.assertIn("Morning thoughts", content_text)  # From 03-27
        self.assertIn("Daily goals", content_text)  # From 03-28
        self.assertIn("Coding session", content_text)  # From 03-29

    def test_load_excludes_files_outside_range(self):
        """Test that files outside the date range are excluded."""
        input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-27", journal_end_date="2025-03-29")

        documents = self.loader.load(input_data)

        # Check that no documents from April 15 are included
        dates = set(doc.metadata["journal_date"] for doc in documents)
        self.assertNotIn("2025-04-15", dates)

        # Check that content from April 15 is not present
        content_text = " ".join([doc.page_content for doc in documents])
        self.assertNotIn("This file should not be included", content_text)

    def test_nested_bullet_points(self):
        """Test that nested bullet points are properly handled."""
        input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-27", journal_end_date="2025-03-27")

        documents = self.loader.load(input_data)

        # Check for content with nested bullet points
        nested_content_found = False
        for doc in documents:
            if "Woke up early today" in doc.page_content or "Discussed project timeline" in doc.page_content:
                nested_content_found = True
                break

        self.assertTrue(nested_content_found, "Nested bullet point content not found")

    def test_code_blocks(self):
        """Test that code blocks are properly handled."""
        input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-29", journal_end_date="2025-03-29")

        documents = self.loader.load(input_data)

        # Check for code block content
        code_block_found = False
        for doc in documents:
            if "def process_data" in doc.page_content:
                code_block_found = True
                break

        self.assertTrue(code_block_found, "Code block content not found")

    def test_tags_in_content(self):
        """Test that Logseq tags are preserved in the content."""
        input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-28", journal_end_date="2025-03-28")

        documents = self.loader.load(input_data)

        # Check for tags in content
        tags_found = False
        for doc in documents:
            if "#planning" in doc.page_content or "#work" in doc.page_content or "#blocked" in doc.page_content:
                tags_found = True
                break

        self.assertTrue(tags_found, "Tags not found in document content")

    def test_references_in_content(self):
        """Test that Logseq references are preserved in the content."""
        input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-27", journal_end_date="2025-03-28")

        documents = self.loader.load(input_data)

        # Check for references in content
        references_found = False
        for doc in documents:
            if "[[Meeting]]" in doc.page_content or "[[Machine Learning]]" in doc.page_content:
                references_found = True
                break

        self.assertTrue(references_found, "References not found in document content")

    def test_document_metadata(self):
        """Test that documents have the correct metadata."""
        input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-27", journal_end_date="2025-03-27")

        documents = self.loader.load(input_data)

        # Check that all documents have the expected metadata fields
        for doc in documents:
            self.assertIn("journal_date", doc.metadata)
            self.assertIn("journal_char_count", doc.metadata)
            self.assertIn("journal_tags", doc.metadata)

            # Verify the values
            self.assertEqual(doc.metadata["journal_date"], "2025-03-27")
            self.assertEqual(doc.metadata["journal_char_count"], len(doc.page_content))
            self.assertIsInstance(doc.metadata["journal_tags"], list)

    def test_boundary_dates(self):
        """Test loading with boundary dates."""
        # Test with start date exactly matching a file
        input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-29", journal_end_date="2025-04-15")

        documents = self.loader.load(input_data)
        dates = set(doc.metadata["journal_date"] for doc in documents)
        self.assertEqual(len(dates), 2)
        self.assertIn("2025-03-29", dates)
        self.assertIn("2025-04-15", dates)

        # Test with end date exactly matching a file
        input_data = LogseqJournalLoaderInput(journal_start_date="2025-03-3", journal_end_date="2025-03-27")

        documents = self.loader.load(input_data)
        dates = set(doc.metadata["journal_date"] for doc in documents)
        self.assertEqual(len(dates), 1)
        self.assertIn("2025-03-27", dates)


if __name__ == "__main__":
    unittest.main()
