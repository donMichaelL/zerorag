from typing import Protocol

from langchain_core.embeddings import Embeddings


class EmbeddingsStrategy(Protocol):
    """Protocol defining the standard interface for all embedding providers."""

    def create(self) -> Embeddings: ...
