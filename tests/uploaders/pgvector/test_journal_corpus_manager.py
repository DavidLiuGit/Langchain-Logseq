import unittest
from unittest.mock import Mock, MagicMock
from uuid import UUID
from pathlib import Path

from pgvector_template.core.embedder import BaseEmbeddingProvider

from langchain_logseq.uploaders.pgvector.journal_corpus_manager import JournalCorpusManager, JournalCorpusManagerConfig


class TestJournalCorpusManagerHelpers(unittest.TestCase):

    def setUp(self):
        mock_session = Mock()
        mock_embedding_provider = Mock(spec=BaseEmbeddingProvider)
        config = JournalCorpusManagerConfig(embedding_provider=mock_embedding_provider)
        self.corpus_manager = JournalCorpusManager(mock_session, config)

    def test_split_corpus_basic(self):
        content = "First line\n- First bullet\n- Second bullet"
        result = self.corpus_manager._split_corpus(content)
        self.assertEqual(result, ["First line", "First bullet", "Second bullet"])

    def test_split_corpus_empty_chunks_filtered(self):
        content = "First line\n-\n- Valid bullet\n-   "
        result = self.corpus_manager._split_corpus(content)
        self.assertEqual(result, ["First line", "Valid bullet"])

    def test_split_corpus_no_bullets(self):
        content = "Just some text without bullets"
        result = self.corpus_manager._split_corpus(content)
        self.assertEqual(result, ["Just some text without bullets"])

    def test_extract_chunk_metadata(self):
        content = "This is a test chunk with five words"
        result = self.corpus_manager._extract_chunk_metadata(content)
        self.assertEqual(
            result,
            {
                "chunk_len": 36,
                "word_count": 8,
                "references": [],
                "anchor_ids": [],
            },
        )

    def test_extract_chunk_metadata_empty(self):
        content = ""
        result = self.corpus_manager._extract_chunk_metadata(content)
        self.assertEqual(
            result,
            {
                "chunk_len": 0,
                "word_count": 0,
                "references": [],
                "anchor_ids": [],
            },
        )

    def test_split_corpus_only_bullets(self):
        content = "- First bullet\n- Second bullet"
        result = self.corpus_manager._split_corpus(content)
        self.assertEqual(result, ["First bullet", "Second bullet"])

    def test_split_corpus_empty_string(self):
        content = ""
        result = self.corpus_manager._split_corpus(content)
        self.assertEqual(result, [])

    def test_split_corpus_whitespace_only(self):
        content = "   \n\t  "
        result = self.corpus_manager._split_corpus(content)
        self.assertEqual(result, [])

    def test_extract_chunk_references_basic(self):
        split_content = ["this", "is", "#my", "test", "#script"]
        result = self.corpus_manager._extract_chunk_references(split_content)
        self.assertEqual(result, ["my", "script"])

    def test_extract_chunk_references_complex(self):
        split_content = ["this", "is", "#my", "test", "#script'sfatal", "flaw", "#lol#"]
        result = self.corpus_manager._extract_chunk_references(split_content)
        self.assertEqual(result, ["my", "script", "lol"])

    def test_extract_chunk_references_none(self):
        split_content = ["no", "references", "here"]
        result = self.corpus_manager._extract_chunk_references(split_content)
        self.assertEqual(result, [])

    def test_extract_chunk_references_empty(self):
        split_content = []
        result = self.corpus_manager._extract_chunk_references(split_content)
        self.assertEqual(result, [])

    def test_extract_chunk_references_dates(self):
        split_content = ["met", "with", "#2025-07-07", "about", "#cookout"]
        result = self.corpus_manager._extract_chunk_references(split_content)
        self.assertEqual(result, ["2025-07-07", "cookout"])

    def test_extract_chunk_metadata_with_references(self):
        content = "Met with team about #project-alpha and #2025-01-15"
        result = self.corpus_manager._extract_chunk_metadata(content)
        self.assertEqual(
            result,
            {
                "chunk_len": 50,
                "word_count": 7,
                "references": ["project-alpha", "2025-01-15"],
                "anchor_ids": [],
            },
        )

    def test_extract_chunk_metadata_special_chars(self):
        content = "Check #bug-fix! and #feature@v2 (important)"
        result = self.corpus_manager._extract_chunk_metadata(content)
        self.assertEqual(
            result,
            {
                "chunk_len": 43,
                "word_count": 5,
                "references": ["bug-fix", "feature@v2"],
                "anchor_ids": [],
            },
        )

    def test_extract_chunk_metadata_multiple_spaces(self):
        content = "Review   #docs    with   #team-lead   tomorrow"
        result = self.corpus_manager._extract_chunk_metadata(content)
        self.assertEqual(
            result,
            {
                "chunk_len": 46,
                "word_count": 5,
                "references": ["docs", "team-lead"],
                "anchor_ids": [],
            },
        )

    def test_extract_chunk_metadata_newlines_tabs(self):
        content = "Task\t#urgent\nfollow up #meeting-notes"
        result = self.corpus_manager._extract_chunk_metadata(content)
        self.assertEqual(
            result,
            {
                "chunk_len": 37,  # note: special chars like \t & \n only count as 1 char each
                "word_count": 5,
                "references": ["urgent", "meeting-notes"],
                "anchor_ids": [],
            },
        )

    def test_extract_chunk_references_breaking_chars(self):
        split_content = ["#asdf?qwer", "#test!end", "#name:value", "#quote'break", '#double"quote']
        result = self.corpus_manager._extract_chunk_references(split_content)
        self.assertEqual(result, ["asdf", "test", "name", "quote", "double"])

    def test_extract_chunk_references_backslash_ignored(self):
        split_content = ["#asdf\\qwer", "#test\\\\path", "#name\\value"]
        result = self.corpus_manager._extract_chunk_references(split_content)
        self.assertEqual(result, ["asdfqwer", "testpath", "namevalue"])

    def test_extract_chunk_references_mixed_special_chars(self):
        split_content = ["#tag\\with?break", "#path\\to:file", "#name'with\"quotes"]
        result = self.corpus_manager._extract_chunk_references(split_content)
        self.assertEqual(result, ["tagwith", "pathto", "name"])

    def test_extract_chunk_references_multiple_hashes(self):
        split_content = ["###ref", "##tag", "####multiple"]
        result = self.corpus_manager._extract_chunk_references(split_content)
        self.assertEqual(result, ["ref", "tag", "multiple"])

    def test_extract_chunk_references_trailing_hashes(self):
        split_content = ["#lol#", "#tag##", "#ref###"]
        result = self.corpus_manager._extract_chunk_references(split_content)
        self.assertEqual(result, ["lol", "tag", "ref"])

    def test_extract_anchor_ids_single(self):
        content = "Some text\n  id:: 686f4ac0-e43b-4a15-940a-954f55e03bea\nMore text"
        result = self.corpus_manager._extract_anchor_ids(content)
        self.assertEqual(result, ["686f4ac0-e43b-4a15-940a-954f55e03bea"])

    def test_extract_anchor_ids_multiple(self):
        content = "First id:: 686f4ac0-e43b-4a15-940a-954f55e03bea\nSecond id:: 12345678-1234-1234-1234-123456789abc"
        result = self.corpus_manager._extract_anchor_ids(content)
        self.assertEqual(result, ["686f4ac0-e43b-4a15-940a-954f55e03bea", "12345678-1234-1234-1234-123456789abc"])

    def test_extract_anchor_ids_none(self):
        content = "No anchor IDs here, just regular text"
        result = self.corpus_manager._extract_anchor_ids(content)
        self.assertEqual(result, [])

    def test_extract_anchor_ids_with_references(self):
        content = "Text with ((686f4ac0-e43b-4a15-940a-954f55e03bea)) reference but id:: 12345678-1234-1234-1234-123456789abc anchor"
        result = self.corpus_manager._extract_anchor_ids(content)
        self.assertEqual(result, ["12345678-1234-1234-1234-123456789abc"])

    def test_extract_chunk_metadata_with_anchor_ids(self):
        content = "Task with #tag\n  id:: 686f4ac0-e43b-4a15-940a-954f55e03bea"
        result = self.corpus_manager._extract_chunk_metadata(content)
        self.assertEqual(
            result,
            {
                "chunk_len": 58,
                "word_count": 5,
                "references": ["tag"],
                "anchor_ids": ["686f4ac0-e43b-4a15-940a-954f55e03bea"],
            },
        )


class TestJournalCorpusManagerE2E(unittest.TestCase):

    def setUp(self):
        self.mock_session = MagicMock()
        self.mock_embedding_provider = Mock(spec=BaseEmbeddingProvider)
        self.mock_embedding_provider.embed_batch.return_value = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024, [0.4] * 1024]

        config = JournalCorpusManagerConfig(embedding_provider=self.mock_embedding_provider)
        self.corpus_manager = JournalCorpusManager(self.mock_session, config)

        # Load real journal content from test file
        test_file_path = Path(__file__).parent.parent.parent / "loaders" / "test_journals" / "2025_07_09.md"
        with open(test_file_path, "r") as f:
            self.journal_content = f.read()

    def test_insert_corpus_e2e(self):
        """Test full corpus insertion workflow with real journal content"""
        corpus_metadata = {"date_str": "2025-07-09"}

        result = self.corpus_manager.insert_corpus(self.journal_content, corpus_metadata)

        # Should return number of documents inserted
        self.assertEqual(result, 4)

        # Verify embedding provider was called with correct chunks
        self.mock_embedding_provider.embed_batch.assert_called_once()
        embedded_chunks = self.mock_embedding_provider.embed_batch.call_args[0][0]
        self.assertEqual(len(embedded_chunks), 4)

        # Verify session operations
        self.mock_session.add_all.assert_called_once()
        self.mock_session.commit.assert_called_once()

        # Check that documents were created with correct structure
        added_docs = self.mock_session.add_all.call_args[0][0]
        self.assertEqual(len(added_docs), 4)

        # Verify first document has expected content and metadata
        first_doc = added_docs[0]
        self.assertIn("cooked", first_doc.content)
        self.assertEqual(first_doc.chunk_index, 0)

    # def test_get_full_corpus_e2e(self):
    #     """Test corpus reconstruction from chunks"""
    #     # Mock database query results
    #     mock_chunks = []
    #     chunks_content = self.corpus_manager._split_corpus(self.journal_content)

    #     for i, content in enumerate(chunks_content):
    #         mock_chunk = Mock()
    #         mock_chunk.chunk_index = i
    #         mock_chunk.content = content
    #         mock_chunk.original_id = "2025-07-09"
    #         mock_chunk.document_metadata = {"date_str": "2025-07-09", "chunk_len": len(content)}
    #         mock_chunk.id = f"chunk_{i}"
    #         mock_chunk.title = None
    #         mock_chunks.append(mock_chunk)

    #     # Mock the query chain
    #     query_mock = Mock()
    #     filter_mock = Mock()
    #     order_mock = Mock()

    #     self.mock_session.query.return_value = query_mock
    #     query_mock.filter.return_value = filter_mock
    #     filter_mock.order_by.return_value = order_mock
    #     order_mock.all.return_value = mock_chunks

    #     result = self.corpus_manager.get_full_corpus("2025-07-09")

    #     self.assertIsNotNone(result)
    #     self.assertEqual(result["id"], "2025-07-09")
    #     self.assertIn("cooked", result["content"])
    #     self.assertIn("pickleball", result["content"])
    #     self.assertEqual(len(result["chunks"]), 5)

    def test_insert_documents_e2e(self):
        """Test direct document insertion"""
        corpus_id = UUID("12345678-1234-1234-1234-123456789abc")
        chunks = self.corpus_manager._split_corpus(self.journal_content)
        embeddings = [[0.1] * 1024] * len(chunks)
        corpus_metadata = {"date_str": "2025-07-09"}

        result = self.corpus_manager.insert_documents(corpus_id, chunks, embeddings, corpus_metadata)

        # empty top-level bullet should be excluded
        self.assertEqual(result, 4)
        self.mock_session.add_all.assert_called_once()
        self.mock_session.commit.assert_called_once()

        added_docs = self.mock_session.add_all.call_args[0][0]
        self.assertEqual(len(added_docs), 4)

        # Verify metadata extraction worked correctly
        first_doc = added_docs[0]
        self.assertIn("cooked", first_doc.content)

        # Find document with references
        ref_doc = next((doc for doc in added_docs if "pickleball" in doc.content), None)
        self.assertIsNotNone(ref_doc)

        # Find document with anchor ID
        anchor_doc = next((doc for doc in added_docs if "686f4ac0-e43b-4a15-940a-954f55e03bea" in doc.content), None)
        self.assertIsNotNone(anchor_doc)
