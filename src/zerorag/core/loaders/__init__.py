import logging
from pathlib import Path

from langchain_core.documents import Document

from .base import DocumentLoaderStrategy
from .docx import DocxDirectoryLoader
from .pdf import PDFDirectoryLoader
from .text import TextDirectoryLoader

logger = logging.getLogger(__name__)


LOADER_REGISTRY: dict[str, type[DocumentLoaderStrategy]] = {
    "txt": TextDirectoryLoader,
    "pdf": PDFDirectoryLoader,
    "docx": DocxDirectoryLoader,
}


def load_documents(source: Path, types: list[str]) -> list[Document]:
    """
    Recursively scan the source for specified file types and return their
    contents as a list of LangChain Document objects.

    Args:
        source (Path): The root path to scan for documents.
        types (list[str]): A list of file extensions to load (e.g., ['txt', 'pdf']).

    Returns:
        list[Document]: A list of Document objects.
    """

    documents: list[Document] = []

    for file_type in types:
        loader_cls = LOADER_REGISTRY.get(file_type)
        if not loader_cls:
            logger.warning(f"Skipping '{file_type}': Unsupported type")
            continue

        logger.debug(f"Scanning for .{file_type} files in {source.absolute()}...")

        loader = loader_cls()
        loaded_docs = loader.load(source)
        logger.debug(f"Loaded {len(loaded_docs)} pages from .{file_type} files")
        documents.extend(loaded_docs)

    return documents


__all__ = ["load_documents"]
