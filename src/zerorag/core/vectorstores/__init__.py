import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .base import VectorStoreStrategy
from .chromadb import ChromaDBVectorStoreBackend
from .inmemory import InMemoryVectorStoreBackend

logger = logging.getLogger(__name__)


VECTORSTORE_REGISTRY: dict[str, type[VectorStoreStrategy]] = {
    "inmemory": InMemoryVectorStoreBackend,
    "chromadb": ChromaDBVectorStoreBackend,
}


def store_documents(
    documents: list[Document],
    embeddings: Embeddings,
    store_dir: Path,
    strategy: str = "inmemory",
) -> None:
    """
    Embed and store documents in a vector store.

    Args:
        documents: The chunked documents to store.
        embeddings: A LangChain Embeddings instance for generating vectors.
        store_dir: Directory to persist the vector store.
        strategy: The vector store backend to use (must be a key in VECTORSTORE_REGISTRY).
    """
    backend_cls = VECTORSTORE_REGISTRY.get(strategy)
    if not backend_cls:
        logger.warning(f"Unknown vector store strategy '{strategy}', falling back to 'inmemory'")
        backend_cls = VECTORSTORE_REGISTRY["inmemory"]

    backend = backend_cls()
    backend.store(documents, embeddings, store_dir)
    logger.debug(f"Stored {len(documents)} documents in '{strategy}' vector store at {store_dir}")


__all__ = ["store_documents"]
