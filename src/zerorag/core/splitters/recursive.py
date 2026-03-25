from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveCharacterSplitter:
    """Splits documents into chunks using recursive character splitting."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 300) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            keep_separator=True,
        )
        return splitter.split_documents(documents)
