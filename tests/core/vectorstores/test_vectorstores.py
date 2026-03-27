from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from zerorag.core.vectorstores import load_vectorstore, store_documents


class TestStoreDocumentsEngine:
    """Test suite for the main registry-based vector store engine."""

    @pytest.mark.parametrize(
        ("strategy", "description"),
        [
            ("inmemory", "explicit inmemory"),
            ("nonexistent", "unknown falls back to inmemory"),
            (None, "default strategy"),
        ],
    )
    @patch("zerorag.core.vectorstores.inmemory.InMemoryVectorStore")
    def test_inmemory_backend_is_used(self, mock_vs_cls, strategy, description, tmp_path):  # noqa: ARG002
        """Test that the inmemory backend is used for: {description}."""
        mock_vs_cls.from_documents.return_value = MagicMock()
        docs = [Document(page_content="hello")]
        kwargs = {"strategy": strategy} if strategy is not None else {}

        store_documents(docs, MagicMock(spec=Embeddings), tmp_path, **kwargs)

        mock_vs_cls.from_documents.assert_called_once()

    def test_explicit_chromadb_strategy(self, tmp_path):
        """Test that explicitly requesting chromadb delegates to Chroma.from_documents."""
        with patch("zerorag.core.vectorstores.chromadb.ChromaDBVectorStoreBackend.store") as mock_store:
            docs = [Document(page_content="hello")]
            embeddings = MagicMock(spec=Embeddings)
            store_documents(docs, embeddings, tmp_path, strategy="chromadb")

            mock_store.assert_called_once_with(docs, embeddings, tmp_path)


class TestLoadVectorstoreEngine:
    """Test suite for the registry-based vector store loading engine."""

    @patch("zerorag.core.vectorstores.inmemory.InMemoryVectorStore")
    def test_load_uses_inmemory_by_default(self, mock_vs_cls, tmp_path):
        """Test that load_vectorstore uses the inmemory backend by default."""
        mock_vs_cls.load.return_value = MagicMock()
        embeddings = MagicMock(spec=Embeddings)

        load_vectorstore(embeddings, tmp_path)

        mock_vs_cls.load.assert_called_once()

    @patch("zerorag.core.vectorstores.inmemory.InMemoryVectorStore")
    def test_load_unknown_strategy_falls_back_to_inmemory(self, mock_vs_cls, tmp_path):
        """Test that an unknown strategy falls back to inmemory."""
        mock_vs_cls.load.return_value = MagicMock()
        embeddings = MagicMock(spec=Embeddings)

        load_vectorstore(embeddings, tmp_path, strategy="nonexistent")

        mock_vs_cls.load.assert_called_once()

    def test_load_explicit_chromadb_strategy(self, tmp_path):
        """Test that explicitly requesting chromadb delegates to ChromaDB load."""
        with patch("zerorag.core.vectorstores.chromadb.ChromaDBVectorStoreBackend.load") as mock_load:
            mock_load.return_value = MagicMock()
            embeddings = MagicMock(spec=Embeddings)

            load_vectorstore(embeddings, tmp_path, strategy="chromadb")

            mock_load.assert_called_once_with(embeddings, tmp_path)
