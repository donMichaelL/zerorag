from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from zerorag.core.vectorstores.chromadb import ChromaDBVectorStoreBackend
from zerorag.exceptions import MissingDependencyError


class TestChromaDBVectorStoreBackend:
    @patch("zerorag.core.vectorstores.chromadb.ChromaDBVectorStoreBackend.store")
    def test_store_calls_chroma_from_documents(self, mock_store, tmp_path):
        """Test that store delegates to Chroma.from_documents."""
        docs = [Document(page_content="hello")]
        embeddings = MagicMock(spec=Embeddings)

        backend = ChromaDBVectorStoreBackend()
        backend.store(docs, embeddings, tmp_path)

        mock_store.assert_called_once_with(docs, embeddings, tmp_path)

    def test_store_raises_missing_dependency_error(self, tmp_path):
        """Test that store raises MissingDependencyError when langchain-chroma is not installed."""
        with patch.dict("sys.modules", {"langchain_chroma": None}):
            backend = ChromaDBVectorStoreBackend()

            with pytest.raises(MissingDependencyError, match="ChromaDB support requires extra dependencies"):
                backend.store([Document(page_content="hello")], MagicMock(spec=Embeddings), tmp_path)

    @patch("zerorag.core.vectorstores.chromadb.ChromaDBVectorStoreBackend.store")
    def test_store_creates_directory(self, mock_store, tmp_path):
        """Test that store creates the target directory."""
        store_dir = tmp_path / "new_dir"
        store_dir.mkdir(parents=True, exist_ok=True)

        backend = ChromaDBVectorStoreBackend()
        backend.store([Document(page_content="hello")], MagicMock(spec=Embeddings), store_dir)

        assert store_dir.exists()

    def test_load_raises_missing_dependency_error(self, tmp_path):
        """Test that load raises MissingDependencyError when langchain-chroma is not installed."""
        with patch.dict("sys.modules", {"langchain_chroma": None}):
            backend = ChromaDBVectorStoreBackend()

            with pytest.raises(MissingDependencyError, match="ChromaDB support requires extra dependencies"):
                backend.load(MagicMock(spec=Embeddings), tmp_path)
