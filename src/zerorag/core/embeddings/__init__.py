import logging

from langchain_core.embeddings import Embeddings

from .base import EmbeddingsStrategy
from .fastembed import FastEmbedProvider
from .openai import OpenAIEmbedProvider

logger = logging.getLogger(__name__)


EMBEDDER_REGISTRY: dict[str, tuple[type[EmbeddingsStrategy], dict]] = {
    "fastembed": (FastEmbedProvider, {}),
    "openai-small": (OpenAIEmbedProvider, {"model_name": "text-embedding-3-small"}),
    "openai-large": (OpenAIEmbedProvider, {"model_name": "text-embedding-3-large"}),
}


def get_embeddings(strategy: str = "fastembed") -> Embeddings:
    """
    Create and return a LangChain Embeddings instance.

    Args:
        strategy: The embedding provider to use (must be a key in EMBEDDER_REGISTRY).

    Returns:
        A LangChain Embeddings instance ready to be passed to a vector store.
    """
    entry = EMBEDDER_REGISTRY.get(strategy)
    if not entry:
        logger.warning(f"Unknown embedder strategy '{strategy}', falling back to 'fastembed'")
        entry = EMBEDDER_REGISTRY["fastembed"]

    provider_cls, kwargs = entry
    return provider_cls(**kwargs).create()


__all__ = ["get_embeddings"]
