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
        self.assertEqual(result, ["First line", " First bullet", " Second bullet"])
    
    def test_split_corpus_empty_chunks_filtered(self):
        content = "First line\n-\n- Valid bullet\n-   "
        result = self.corpus_manager._split_corpus(content)
        self.assertEqual(result, ["First line", " Valid bullet"])
    
    def test_split_corpus_no_bullets(self):
        content = "Just some text without bullets"
        result = self.corpus_manager._split_corpus(content)
        self.assertEqual(result, ["Just some text without bullets"])
    
    def test_extract_chunk_metadata(self):
        content = "This is a test chunk with five words"
        result = self.corpus_manager._extract_chunk_metadata(content)
        self.assertEqual(result, {
            "chunk_len": 36,
            "word_count": 8,
        })
    
    def test_extract_chunk_metadata_empty(self):
        content = ""
        result = self.corpus_manager._extract_chunk_metadata(content)
        self.assertEqual(result, {
            "chunk_len": 0,
            "word_count": 0,
        })