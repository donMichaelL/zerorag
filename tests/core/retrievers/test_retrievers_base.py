from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class MockRetrieverStrategy:
    def retrieve(self, query: str, vector_store: VectorStore, **kwargs: Any) -> list[Document]:
        return [Document(page_content="mock result")]


def test_retriever_strategy_contract():
    """
    Test that a custom class implementing the RetrieverStrategy
    protocol successfully fulfills the required contract.
    """
    strategy = MockRetrieverStrategy()
    results = strategy.retrieve("test query", vector_store=None)

    assert len(results) == 1
    assert isinstance(results[0], Document)
