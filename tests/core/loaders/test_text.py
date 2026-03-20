from pathlib import Path

import pytest

from zerorag.core.loaders.text import TextDirectoryLoader


@pytest.fixture
def loader() -> TextDirectoryLoader:
    return TextDirectoryLoader()


@pytest.fixture
def base_txt_dir(tmp_path: Path) -> Path:
    (tmp_path / "doc1.txt").write_text("Content 1")
    (tmp_path / "doc2.txt").write_text("Content 2")
    return tmp_path


@pytest.fixture
def dir_with_nested_txt(base_txt_dir: Path) -> Path:
    sub_dir = base_txt_dir / "nested"
    sub_dir.mkdir()
    (sub_dir / "deep.txt").write_text("Nested content")
    return base_txt_dir


@pytest.fixture
def dir_with_mixed_files(base_txt_dir: Path) -> Path:
    (base_txt_dir / "ignored.csv").write_text("id,val\n1,2")
    return base_txt_dir


class TestTextDirectoryLoader:
    def test_loads_base_txt_files(self, loader: TextDirectoryLoader, base_txt_dir: Path):
        """Test that the loader correctly identifies and loads standard text files in the root directory."""
        documents = loader.load(base_txt_dir)

        assert len(documents) == 2

    def test_loads_nested_txt_files(self, loader: TextDirectoryLoader, dir_with_nested_txt: Path):
        """Test that the loader recursively finds and loads text files hidden inside subdirectories."""
        documents = loader.load(dir_with_nested_txt)

        assert len(documents) == 3

    def test_ignores_non_txt_files(self, loader: TextDirectoryLoader, dir_with_mixed_files: Path):
        """Test that the loader strictly ignores files that do not match the .txt extension."""
        documents = loader.load(dir_with_mixed_files)
        sources = [doc.metadata.get("source", "") for doc in documents]

        assert len(documents) == 2
        assert not any("ignored.csv" in source for source in sources)

    def test_empty_directory(self, loader: TextDirectoryLoader, tmp_path: Path):
        """Test that the loader gracefully returns an empty list when scanning an empty directory."""
        documents = loader.load(tmp_path)

        assert isinstance(documents, list)
        assert len(documents) == 0
