import unittest
from unittest.mock import Mock
from pgvector_template.core.embedder import BaseEmbeddingProvider
from langchain_logseq.uploaders.pgvector.journal_corpus_manager import JournalCorpusManager, JournalCorpusManagerConfig


class TestJournalCorpusManager(unittest.TestCase):

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
