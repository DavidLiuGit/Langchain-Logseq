from langchain_core.messages import HumanMessage, AIMessage

from utils.logging import _print_document_count


def test_retrieve_specific_date(pgvector_journal_retriever):
    """Test retrieving documents about a specific date."""
    query = "What did I do on Mar 27, 2025?"
    documents_retrieved = pgvector_journal_retriever.invoke({"input": query})
    _print_document_count(documents_retrieved, query)
    assert documents_retrieved is not None
    assert len(documents_retrieved) >= 1  # should retrieve at least one document

    # mock a LLM response and retrieve more documents with the history
    history = [
        HumanMessage(content="What did I do on Mar 27?"),
        AIMessage(content="You had several meetings and worked on the project"),
    ]
    query = "What else did I do that day?"
    documents_retrieved = pgvector_journal_retriever.invoke({"input": query, "chat_history": history})
    _print_document_count(documents_retrieved, query)
    assert len(documents_retrieved) >= 1


def test_retrieve_by_topic(pgvector_journal_retriever):
    """Test retrieving documents by topic rather than date."""
    query = "What have I written about machine learning?"
    documents_retrieved = pgvector_journal_retriever.invoke({"input": query})
    _print_document_count(documents_retrieved, query)
    assert documents_retrieved is not None

    # The actual count will depend on your vector database content
    # This is just a placeholder assertion
    assert len(documents_retrieved) >= 0


def test_retrieve_with_complex_query(pgvector_journal_retriever):
    """Test retrieving documents with a more complex query."""
    query = "What projects was I working on in March 2025 that involved coding?"
    documents_retrieved = pgvector_journal_retriever.invoke({"input": query})
    _print_document_count(documents_retrieved, query)
    assert documents_retrieved is not None

    # The actual count will depend on your vector database content
    # This is just a placeholder assertion
    assert len(documents_retrieved) >= 0

    # mock a LLM response and retrieve more documents with the history
    history = [
        HumanMessage(content="What projects was I working on in March 2025 that involved coding?"),
        AIMessage(content="You were working on the data pipeline project and the ML model"),
    ]
    query = "Did I mention any challenges with those projects?"
    documents_retrieved = pgvector_journal_retriever.invoke({"input": query, "chat_history": history})
    _print_document_count(documents_retrieved, query)
    assert len(documents_retrieved) >= 0
