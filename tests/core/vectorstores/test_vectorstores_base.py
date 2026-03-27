from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class MockVectorStoreStrategy:
    def store(self, documents: list[Document], embeddings: Embeddings, store_dir: Path) -> None:  # noqa: ARG002
        pass

    def load(self, embeddings: Embeddings, store_dir: Path) -> VectorStore:  # noqa: ARG002
        return MagicMock(spec=VectorStore)


def test_vector_store_strategy_store_contract(tmp_path):
    """
    Test that a custom class implementing the VectorStoreStrategy
    protocol successfully fulfills the store contract.
    """
    strategy = MockVectorStoreStrategy()
    documents = [Document(page_content="some content")]
    embeddings = MagicMock(spec=Embeddings)

    strategy.store(documents, embeddings, tmp_path)


def test_vector_store_strategy_load_contract(tmp_path):
    """
    Test that a custom class implementing the VectorStoreStrategy
    protocol successfully fulfills the load contract.
    """
    strategy = MockVectorStoreStrategy()
    embeddings = MagicMock(spec=Embeddings)

    result = strategy.load(embeddings, tmp_path)

    assert isinstance(result, VectorStore)
