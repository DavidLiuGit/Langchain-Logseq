import unittest
from unittest.mock import Mock
from pgvector_template.core.document import BaseDocumentOptionalProps
from utils.bedrock_embedder import BedrockEmbeddingProvider
from langchain_logseq.uploaders.pgvector.journal_corpus_manager import JournalCorpusManager, JournalCorpusManagerConfig


class TestJournalCorpusManager(unittest.TestCase):
    
    def setUp(self):
        self.mock_session = Mock()
        self.embedding_provider = BedrockEmbeddingProvider(verbose=True)
        config = JournalCorpusManagerConfig(embedding_provider=self.embedding_provider)
        self.corpus_manager = JournalCorpusManager(self.mock_session, config)
    
    def test_simple_insert_corpus_integration(self):
        content = "Daily notes\n- Task 1\n- Task 2"
        corpus_metadata = {"date_str": "2024-01-01"}
        optional_props = BaseDocumentOptionalProps()
        
        result = self.corpus_manager.insert_corpus(content, corpus_metadata, optional_props)
        
        self.assertEqual(result, 3)
        self.mock_session.add_all.assert_called_once()
        self.mock_session.commit.assert_called_once()
        
        # Check documents passed to add_all
        added_docs = self.mock_session.add_all.call_args[0][0]
        self.assertEqual(len(added_docs), 3)
        self.assertEqual(added_docs[0].content, "Daily notes")
        self.assertEqual(added_docs[1].content, "Task 1")
        self.assertEqual(added_docs[2].content, "Task 2")
        
        # Verify all docs have same corpus_id
        corpus_id = added_docs[0].corpus_id
        self.assertTrue(all(doc.corpus_id == corpus_id for doc in added_docs))
        
        # Verify chunk indices
        self.assertEqual([doc.chunk_index for doc in added_docs], [0, 1, 2])
