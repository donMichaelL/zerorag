from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


class TextDirectoryLoader:
    """Handles loading of standard .txt files."""

    def load(self, source: Path) -> list[Document]:
        loader = DirectoryLoader(
            path=str(source),
            glob="**/*.txt",
            loader_cls=TextLoader,
            silent_errors=True,
            use_multithreading=True,
        )
        return loader.load()
