from unittest.mock import MagicMock

from langchain_core.documents import Document

from zerorag.core.retrievers.similarity import SimilarityRetriever


class TestSimilarityRetriever:
    def test_retrieve_returns_documents(self):
        """Test that retrieve() returns documents from the vector store."""
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [Document(page_content="result")]

        retriever = SimilarityRetriever()
        results = retriever.retrieve("test query", mock_vector_store, k=3)

        assert len(results) == 1
        assert results[0].page_content == "result"

    def test_retrieve_uses_similarity_search_type(self):
        """Test that retrieve() configures the retriever with similarity search type."""
        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[]))

        retriever = SimilarityRetriever()
        retriever.retrieve("test query", mock_vector_store, k=3)

        mock_vector_store.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 3})

    def test_retrieve_defaults_k_to_five(self):
        """Test that retrieve() defaults to k=5 when not provided."""
        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[]))

        retriever = SimilarityRetriever()
        retriever.retrieve("test query", mock_vector_store)

        mock_vector_store.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 5})

    def test_retrieve_invokes_with_query(self):
        """Test that retrieve() passes the query string to the retriever."""
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []

        retriever = SimilarityRetriever()
        retriever.retrieve("my specific query", mock_vector_store)

        mock_retriever.invoke.assert_called_once_with("my specific query")
