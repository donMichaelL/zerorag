from pathlib import Path
from unittest.mock import patch

import pytest

from zerorag.core.loaders.docx import DocxDirectoryLoader
from zerorag.exceptions import MissingDependencyError

pytestmark = pytest.mark.usefixtures("fast_directory_loader")


@pytest.fixture
def loader() -> DocxDirectoryLoader:
    return DocxDirectoryLoader()


@pytest.fixture
def base_docx_dir(tmp_path: Path) -> Path:
    (tmp_path / "doc1.docx").touch()
    (tmp_path / "doc2.docx").touch()
    return tmp_path


@pytest.fixture
def dir_with_nested_docx(base_docx_dir: Path) -> Path:
    sub_dir = base_docx_dir / "nested"
    sub_dir.mkdir()
    (sub_dir / "deep.docx").touch()
    return base_docx_dir


@pytest.fixture
def dir_with_mixed_files(base_docx_dir: Path) -> Path:
    (base_docx_dir / "ignored.csv").write_text("id,val\n1,2")
    return base_docx_dir


class TestDocxDirectoryLoader:
    def test_loads_base_docx_files(self, loader: DocxDirectoryLoader, base_docx_dir: Path):
        documents = loader.load(base_docx_dir)
        sources = [doc.metadata.get("source", "") for doc in documents]

        assert len(documents) == 2
        assert any("doc1.docx" in source for source in sources)
        assert any("doc2.docx" in source for source in sources)

    def test_loads_nested_docx_files(self, loader: DocxDirectoryLoader, dir_with_nested_docx: Path):
        documents = loader.load(dir_with_nested_docx)

        assert len(documents) == 3
        sources = [doc.metadata.get("source", "") for doc in documents]
        assert any("deep.docx" in source for source in sources)

    def test_ignores_non_docx_files(self, loader: DocxDirectoryLoader, dir_with_mixed_files: Path):
        documents = loader.load(dir_with_mixed_files)

        assert len(documents) == 2
        sources = [doc.metadata.get("source", "") for doc in documents]
        assert not any("ignored.csv" in source for source in sources)

    def test_empty_directory(self, loader: DocxDirectoryLoader, tmp_path: Path):
        documents = loader.load(tmp_path)

        assert isinstance(documents, list)
        assert len(documents) == 0

    @patch("importlib.util.find_spec")
    def test_missing_dependency_raises_error(self, mock_find_spec, loader: DocxDirectoryLoader, tmp_path: Path):
        mock_find_spec.return_value = None

        with pytest.raises(MissingDependencyError) as exc_info:
            loader.load(tmp_path)

        assert "Word support is missing" in str(exc_info.value)
        assert "pip install 'zerorag[docx]'" in str(exc_info.value)
