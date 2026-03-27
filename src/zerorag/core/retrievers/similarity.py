from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class SimilarityRetriever:
    """Retrieves documents using cosine similarity search."""

    def retrieve(self, query: str, vector_store: VectorStore, **kwargs: Any) -> list[Document]:
        k = kwargs.get("k", 5)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return retriever.invoke(query)
