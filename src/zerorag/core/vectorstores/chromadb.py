from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from zerorag.exceptions import MissingDependencyError


class ChromaDBVectorStoreBackend:
    """Stores documents in a ChromaDB collection with directory persistence."""

    def store(self, documents: list[Document], embeddings: Embeddings, store_dir: Path) -> None:
        try:
            from langchain_chroma import Chroma
        except ImportError as err:
            raise MissingDependencyError(
                'ChromaDB support requires extra dependencies. Install them with: pip install "zerorag[chromadb]"'
            ) from err

        store_dir.mkdir(parents=True, exist_ok=True)
        Chroma.from_documents(documents, embedding=embeddings, persist_directory=str(store_dir))
