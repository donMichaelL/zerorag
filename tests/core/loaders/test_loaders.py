from pathlib import Path

from zerorag.core.loaders import load_documents


class TestLoadDocumentsEngine:
    """Test suite for the main registry-based ingestion engine."""

    def test_loads_supported_file_types(self, tmp_path: Path):
        """Test that the engine successfully loads documents of supported file types."""
        (tmp_path / "engine_test.txt").write_text("Engine routing successful.")
        documents = load_documents(tmp_path, types=["txt"])

        assert len(documents) == 1
        assert documents[0].page_content == "Engine routing successful."

    def test_ignores_unsupported_file_types(self, tmp_path: Path):
        """Test that the engine ignores files with unsupported extensions."""
        (tmp_path / "unsupported.exe").write_text("This should be ignored.")
        documents = load_documents(tmp_path, types=["exe"])

        assert len(documents) == 0, "Should ignore unsupported file types"
