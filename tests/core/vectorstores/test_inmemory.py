from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from zerorag.core.vectorstores.inmemory import STORE_FILENAME, InMemoryVectorStoreBackend


class TestInMemoryVectorStoreBackend:
    @patch("zerorag.core.vectorstores.inmemory.InMemoryVectorStore")
    def test_store_creates_directory(self, mock_vs_cls, tmp_path):
        """Test that store creates the target directory if it doesn't exist."""
        store_dir = tmp_path / "new_dir"
        mock_vs_cls.from_documents.return_value = MagicMock()

        backend = InMemoryVectorStoreBackend()
        backend.store([Document(page_content="hello")], MagicMock(spec=Embeddings), store_dir)

        assert store_dir.exists()

    @patch("zerorag.core.vectorstores.inmemory.InMemoryVectorStore")
    def test_store_calls_from_documents(self, mock_vs_cls, tmp_path):
        """Test that store passes documents and embeddings to InMemoryVectorStore.from_documents."""
        docs = [Document(page_content="hello")]
        embeddings = MagicMock(spec=Embeddings)
        mock_vs_cls.from_documents.return_value = MagicMock()

        backend = InMemoryVectorStoreBackend()
        backend.store(docs, embeddings, tmp_path)

        mock_vs_cls.from_documents.assert_called_once_with(docs, embedding=embeddings)

    @patch("zerorag.core.vectorstores.inmemory.InMemoryVectorStore")
    def test_store_dumps_to_correct_path(self, mock_vs_cls, tmp_path):
        """Test that store dumps the vector store to the expected file path."""
        mock_instance = MagicMock()
        mock_vs_cls.from_documents.return_value = mock_instance

        backend = InMemoryVectorStoreBackend()
        backend.store([Document(page_content="hello")], MagicMock(spec=Embeddings), tmp_path)

        expected_path = str(tmp_path / STORE_FILENAME)
        mock_instance.dump.assert_called_once_with(expected_path)

    @patch("zerorag.core.vectorstores.inmemory.InMemoryVectorStore")
    def test_store_creates_nested_directories(self, mock_vs_cls, tmp_path):
        """Test that store creates nested parent directories."""
        store_dir = tmp_path / "a" / "b" / "c"
        mock_vs_cls.from_documents.return_value = MagicMock()

        backend = InMemoryVectorStoreBackend()
        backend.store([Document(page_content="hello")], MagicMock(spec=Embeddings), store_dir)

        assert store_dir.exists()
