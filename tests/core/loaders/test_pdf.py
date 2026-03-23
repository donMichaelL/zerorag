from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

from zerorag.core.loaders.pdf import PDFDirectoryLoader
from zerorag.exceptions import MissingDependencyError


@pytest.fixture(autouse=True)
def fast_directory_loader(monkeypatch) -> None:
    """
    Intercept LangChain's DirectoryLoader to completely bypass the ThreadPoolExecutor.
    This drops the execution time from 8s to ~0.01s while still perfectly validating
    our file routing and nested folder logic.
    """

    def bypass_threadpool_load(self):
        docs = []
        for file_path in Path(self.path).glob(self.glob):
            if file_path.is_file():
                docs.append(Document(page_content="speed test", metadata={"source": str(file_path)}))
        return docs

    monkeypatch.setattr(DirectoryLoader, "load", bypass_threadpool_load)


@pytest.fixture
def loader() -> PDFDirectoryLoader:
    return PDFDirectoryLoader()


@pytest.fixture
def base_pdf_dir(tmp_path: Path) -> Path:
    (tmp_path / "doc1.pdf").touch()
    (tmp_path / "doc2.pdf").touch()
    return tmp_path


@pytest.fixture
def dir_with_nested_pdf(base_pdf_dir: Path) -> Path:
    sub_dir = base_pdf_dir / "nested"
    sub_dir.mkdir()
    (sub_dir / "deep.pdf").touch()
    return base_pdf_dir


@pytest.fixture
def dir_with_mixed_files(base_pdf_dir: Path) -> Path:
    (base_pdf_dir / "ignored.csv").write_text("id,val\n1,2")
    return base_pdf_dir


class TestPDFDirectoryLoader:
    def test_loads_base_pdf_files(self, loader: PDFDirectoryLoader, base_pdf_dir: Path):
        documents = loader.load(base_pdf_dir)
        sources = [doc.metadata.get("source", "") for doc in documents]

        assert len(documents) == 2
        assert any("doc1.pdf" in source for source in sources)
        assert any("doc2.pdf" in source for source in sources)

    def test_loads_nested_pdf_files(self, loader: PDFDirectoryLoader, dir_with_nested_pdf: Path):
        documents = loader.load(dir_with_nested_pdf)

        assert len(documents) == 3
        sources = [doc.metadata.get("source", "") for doc in documents]
        assert any("deep.pdf" in source for source in sources)

    def test_ignores_non_pdf_files(self, loader: PDFDirectoryLoader, dir_with_mixed_files: Path):
        documents = loader.load(dir_with_mixed_files)

        assert len(documents) == 2
        sources = [doc.metadata.get("source", "") for doc in documents]
        assert not any("ignored.csv" in source for source in sources)

    def test_empty_directory(self, loader: PDFDirectoryLoader, tmp_path: Path):
        documents = loader.load(tmp_path)

        assert isinstance(documents, list)
        assert len(documents) == 0

    @patch("importlib.util.find_spec")
    def test_missing_dependency_raises_error(self, mock_find_spec, loader: PDFDirectoryLoader, tmp_path: Path):
        mock_find_spec.return_value = None

        with pytest.raises(MissingDependencyError) as exc_info:
            loader.load(tmp_path)

        assert "PDF support is missing" in str(exc_info.value)
        assert "pip install 'zerorag[pdf]'" in str(exc_info.value)
