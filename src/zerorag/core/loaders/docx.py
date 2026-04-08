import importlib.util
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

from zerorag.exceptions import MissingDependencyError


class DocxDirectoryLoader:
    """Handles loading of .docx files."""

    def load(self, target_path: Path) -> list[Document]:
        if importlib.util.find_spec("docx2txt") is None:
            raise MissingDependencyError("Word support is missing. Run: pip install 'zerorag[docx]'")

        from langchain_community.document_loaders import Docx2txtLoader

        loader = DirectoryLoader(
            path=str(target_path),
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            silent_errors=True,
            show_progress=False,
            use_multithreading=True,
        )

        return loader.load()
