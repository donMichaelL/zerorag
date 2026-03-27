import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .base import RetrieverStrategy
from .similarity import SimilarityRetriever

logger = logging.getLogger(__name__)


RETRIEVER_REGISTRY: dict[str, type[RetrieverStrategy]] = {
    "similarity": SimilarityRetriever,
}


def retrieve_documents(
    query: str,
    vector_store: VectorStore,
    strategy: str = "similarity",
    **kwargs: Any,
) -> list[Document]:
    """
    Retrieve relevant documents from a vector store.

    Args:
        query: The search query.
        vector_store: A loaded LangChain VectorStore instance.
        strategy: The retrieval strategy to use (must be a key in RETRIEVER_REGISTRY).
        **kwargs: Strategy-specific arguments (e.g. k, fetch_k, score_threshold).
    """
    retriever_cls = RETRIEVER_REGISTRY.get(strategy)
    if not retriever_cls:
        logger.warning(f"Unknown retriever strategy '{strategy}', falling back to 'similarity'")
        retriever_cls = RETRIEVER_REGISTRY["similarity"]

    retriever = retriever_cls()
    return retriever.retrieve(query, vector_store, **kwargs)


__all__ = ["retrieve_documents"]
