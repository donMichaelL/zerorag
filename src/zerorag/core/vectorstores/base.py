from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class VectorStoreStrategy(Protocol):
    """Protocol defining the standard interface for all vector store backends."""

    def store(self, documents: list[Document], embeddings: Embeddings, store_dir: Path) -> None: ...

    def load(self, embeddings: Embeddings, store_dir: Path) -> VectorStore: ...
