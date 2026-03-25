from typing import Protocol

from langchain_core.documents import Document


class DocumentSplitterStrategy(Protocol):
    """Protocol defining the standard interface for all document splitters."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None: ...

    def split(self, documents: list[Document]) -> list[Document]: ...
