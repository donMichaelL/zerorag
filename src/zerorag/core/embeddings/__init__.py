import logging

from langchain_core.embeddings import Embeddings

from .base import EmbeddingsStrategy
from .fastembed import FastEmbedProvider

logger = logging.getLogger(__name__)


EMBEDDER_REGISTRY: dict[str, type[EmbeddingsStrategy]] = {
    "fastembed": FastEmbedProvider,
}


def get_embeddings(strategy: str = "fastembed") -> Embeddings:
    """
    Create and return a LangChain Embeddings instance.

    Args:
        strategy: The embedding provider to use (must be a key in EMBEDDER_REGISTRY).

    Returns:
        A LangChain Embeddings instance ready to be passed to a vector store.
    """
    provider_cls = EMBEDDER_REGISTRY.get(strategy)
    if not provider_cls:
        logger.warning(f"Unknown embedder strategy '{strategy}', falling back to 'fastembed'")
        provider_cls = EMBEDDER_REGISTRY["fastembed"]

    return provider_cls().create()


__all__ = ["get_embeddings"]
