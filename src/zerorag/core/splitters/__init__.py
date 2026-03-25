import logging

from langchain_core.documents import Document

from .base import DocumentSplitterStrategy
from .recursive import RecursiveCharacterSplitter

logger = logging.getLogger(__name__)


SPLITTER_REGISTRY: dict[str, type[DocumentSplitterStrategy]] = {
    "recursive": RecursiveCharacterSplitter,
}


def split_documents(
    documents: list[Document],
    strategy: str = "recursive",
    chunk_size: int = 1200,
    chunk_overlap: int = 300,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks.

    Args:
        documents: The documents to split.
        strategy: The splitting strategy to use (must be a key in SPLITTER_REGISTRY).
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of chunked Document objects.
    """
    splitter_cls = SPLITTER_REGISTRY.get(strategy)
    if not splitter_cls:
        logger.warning(f"Unknown splitter strategy '{strategy}', falling back to 'recursive'")
        splitter_cls = SPLITTER_REGISTRY["recursive"]

    splitter = splitter_cls(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")

    return chunks


__all__ = ["split_documents"]
