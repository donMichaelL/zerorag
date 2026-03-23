import importlib.util
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

from zerorag.exceptions import MissingDependencyError


class PDFDirectoryLoader:
    """Handles loading of .pdf files."""

    def load(self, source: Path) -> list[Document]:
        if importlib.util.find_spec("pypdf") is None:
            raise MissingDependencyError("PDF support is missing. Run: pip install 'zerorag[pdf]'")

        from langchain_community.document_loaders import PyPDFLoader

        loader = DirectoryLoader(
            path=str(source),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True,
            show_progress=False,
            use_multithreading=True,
            loader_kwargs={"extract_images": False},
        )
        return loader.load()
