from unittest.mock import MagicMock

from langchain_core.documents import Document

from zerorag.core.retrievers import retrieve_documents


class TestRetrieveDocumentsEngine:
    """Test suite for the main registry-based retriever engine."""

    def test_returns_documents_with_default_strategy(self):
        """Test that the engine returns documents using the default strategy."""
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [Document(page_content="result")]

        results = retrieve_documents("test query", mock_vector_store, k=3)

        assert len(results) == 1
        assert results[0].page_content == "result"

    def test_unknown_strategy_falls_back_to_similarity(self):
        """Test that an unknown strategy name falls back to similarity."""
        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[]))

        results = retrieve_documents("test query", mock_vector_store, strategy="nonexistent", k=3)

        assert results == []
        mock_vector_store.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 3})

    def test_explicit_similarity_strategy(self):
        """Test that explicitly requesting similarity works."""
        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[]))

        retrieve_documents("test query", mock_vector_store, strategy="similarity", k=5)

        mock_vector_store.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 5})

    def test_kwargs_are_passed_through(self):
        """Test that extra kwargs are forwarded to the retriever strategy."""
        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[]))

        retrieve_documents("test query", mock_vector_store, k=10)

        mock_vector_store.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 10})
