from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from zerorag.exceptions import MissingDependencyError


class ChromaDBVectorStoreBackend:
    """Stores documents in a ChromaDB collection with directory persistence."""

    def store(self, documents: list[Document], embeddings: Embeddings, store_dir: Path) -> None:
        try:
            from langchain_chroma import Chroma
        except ImportError as err:
            raise MissingDependencyError("ChromaDB support is missing. Run: pip install 'zerorag[chromadb]'") from err

        store_dir.mkdir(parents=True, exist_ok=True)
        Chroma.from_documents(documents, embedding=embeddings, persist_directory=str(store_dir))

    def load(self, embeddings: Embeddings, store_dir: Path) -> VectorStore:
        try:
            from langchain_chroma import Chroma
        except ImportError as err:
            raise MissingDependencyError("ChromaDB support is missing. Run: pip install 'zerorag[chromadb]'") from err

        return Chroma(embedding_function=embeddings, persist_directory=str(store_dir))
