from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class RetrieverStrategy(Protocol):
    """Protocol defining the standard interface for all retriever backends."""

    def retrieve(self, query: str, vector_store: VectorStore, **kwargs: Any) -> list[Document]: ...
