from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

STORE_FILENAME = "vectorstore.json"


class InMemoryVectorStoreBackend:
    """Stores documents in a LangChain InMemoryVectorStore and persists to disk."""

    def store(self, documents: list[Document], embeddings: Embeddings, store_dir: Path) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        vector_store = InMemoryVectorStore.from_documents(documents, embedding=embeddings)
        vector_store.dump(str(store_dir / STORE_FILENAME))
