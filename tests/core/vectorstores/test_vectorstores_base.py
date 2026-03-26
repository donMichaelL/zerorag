from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class MockVectorStoreStrategy:
    def store(self, documents: list[Document], embeddings: Embeddings, store_dir: Path) -> None:  # noqa: ARG002
        pass


def test_vector_store_strategy_contract(tmp_path):
    """
    Test that a custom class implementing the VectorStoreStrategy
    protocol successfully fulfills the required contract.
    """
    strategy = MockVectorStoreStrategy()
    documents = [Document(page_content="some content")]
    embeddings = MagicMock(spec=Embeddings)

    strategy.store(documents, embeddings, tmp_path)
