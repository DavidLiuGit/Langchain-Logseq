import pytest
from unittest.mock import Mock
from pgvector_template.core.document import BaseDocumentOptionalProps
from utils.bedrock_embedder import BedrockEmbeddingProvider
from langchain_logseq.uploaders.pgvector.journal_corpus_manager import JournalCorpusManager, JournalCorpusManagerConfig


@pytest.fixture
def corpus_manager():
    mock_session = Mock()
    embedding_provider = BedrockEmbeddingProvider(verbose=True)
    config = JournalCorpusManagerConfig(embedding_provider=embedding_provider)
    return JournalCorpusManager(mock_session, config)


def test_simple_insert_corpus_integration(corpus_manager):
    content = "Daily notes\n- Task 1\n- Task 2"
    corpus_metadata = {"date_str": "2024-01-01"}
    optional_props = BaseDocumentOptionalProps()
    
    result = corpus_manager.insert_corpus(content, corpus_metadata, optional_props)
    
    assert result == 3
    corpus_manager.session.add_all.assert_called_once()
    corpus_manager.session.commit.assert_called_once()
    
    # Check documents passed to add_all
    added_docs = corpus_manager.session.add_all.call_args[0][0]
    assert len(added_docs) == 3
    assert added_docs[0].content == "Daily notes"
    assert added_docs[1].content == "Task 1"
    assert added_docs[2].content == "Task 2"
    
    # Verify all docs have same corpus_id
    corpus_id = added_docs[0].corpus_id
    assert all(doc.corpus_id == corpus_id for doc in added_docs)
    
    # Verify chunk indices
    assert [doc.chunk_index for doc in added_docs] == [0, 1, 2]
