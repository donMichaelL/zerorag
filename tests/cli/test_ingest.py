from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from langchain_core.documents import Document

from zerorag.cli.ingest import ingest, parse_comma_separated_types


class TestParseCommaSeparatedTypes:
    """Unit tests for the CLI string sanitization callback."""

    def test_empty_string_returns_default(self) -> None:
        """Test that empty inputs fallback to the 'txt' default."""
        assert parse_comma_separated_types(None, None, "") == ["txt"]
        assert parse_comma_separated_types(None, None, None) == ["txt"]

    def test_cleans_spaces_and_casing(self) -> None:
        """Test that the parser handles terrible human typing."""
        raw_input = "TXT, pdf,  Docx "
        expected = ["txt", "pdf", "docx"]

        assert parse_comma_separated_types(None, None, raw_input) == expected


class TestIngestCommand:
    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a standard Click CLI runner for the test class."""
        return CliRunner()

    def test_ingest_missing_source_fails(self, runner: CliRunner) -> None:
        """Test that omitting the required source argument throws an error."""
        result = runner.invoke(ingest)

        assert result.exit_code == 2
        assert "Missing argument 'SOURCE'" in result.output

    def test_ingest_nonexistent_directory_fails(self, runner: CliRunner) -> None:
        """Test that providing a path that doesn't exist triggers Click's validation."""
        result = runner.invoke(ingest, ["./ghost_folder"])

        assert result.exit_code == 2
        assert "Directory './ghost_folder' does not exist" in result.output

    def test_ingest_file_instead_of_directory_fails(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that providing a file instead of a folder triggers Click's file_okay=False."""
        test_file = tmp_path / "document.txt"
        test_file.touch()
        result = runner.invoke(ingest, [str(test_file)])

        assert result.exit_code == 2
        assert "is a file" in result.output

    def test_ingest_empty_directory_shows_warning(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that a valid but empty directory shows a warning and exits early."""
        result = runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Initializing ZeroRAG Ingestion" in result.output
        assert "No pages found" in result.output

    def test_ingest_default_types(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that omitting --types correctly prints the default."""
        result = runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Formats   : [txt]" in result.output

    def test_ingest_custom_types(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test passing custom types with messy spacing and casing."""
        result = runner.invoke(ingest, [str(tmp_path), "--types", "TXT, pdf,  Csv "])

        assert "Formats   : [txt, pdf, csv]" in result.output

    def test_ingest_default_store_options(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that default store options are displayed correctly."""
        result = runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Store     : inmemory" in result.output
        assert "Store Dir :" in result.output

    def test_ingest_invalid_store_strategy_fails(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that an invalid --store value is rejected by Click's Choice."""
        result = runner.invoke(ingest, [str(tmp_path), "--store", "invalid"])

        assert result.exit_code == 2
        assert "Invalid value" in result.output

    @patch("zerorag.cli.ingest.store_documents")
    @patch("zerorag.cli.ingest.get_embeddings")
    @patch("zerorag.cli.ingest.split_documents")
    @patch("zerorag.cli.ingest.load_documents")
    def test_ingest_full_pipeline(
        self, mock_load, mock_split, mock_embed, mock_store, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test the full pipeline with mocked core functions."""
        mock_load.return_value = [Document(page_content="hello")]
        mock_split.return_value = [Document(page_content="hello")]
        mock_embed.return_value = MagicMock()

        result = runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Loading documents" in result.output
        assert "Chunking documents" in result.output
        assert "Initializing embeddings" in result.output
        assert "Storing vectors" in result.output
        assert "Ingestion complete" in result.output
        mock_load.assert_called_once()
        mock_split.assert_called_once()
        mock_embed.assert_called_once()
        mock_store.assert_called_once()

    @patch("zerorag.cli.ingest.store_documents")
    @patch("zerorag.cli.ingest.get_embeddings")
    @patch("zerorag.cli.ingest.split_documents")
    @patch("zerorag.cli.ingest.load_documents")
    def test_ingest_passes_store_strategy(
        self, mock_load, mock_split, mock_embed, mock_store, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that --store flag is passed through to store_documents."""
        mock_load.return_value = [Document(page_content="hello")]
        mock_split.return_value = [Document(page_content="hello")]
        mock_embed.return_value = MagicMock()

        result = runner.invoke(ingest, [str(tmp_path), "--store", "chromadb"])

        assert result.exit_code == 0
        assert "Store     : chromadb" in result.output
        mock_store.assert_called_once()
        call_kwargs = mock_store.call_args
        assert call_kwargs[1]["strategy"] == "chromadb"

    def test_ingest_default_embeddings(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that default embeddings strategy is displayed correctly."""
        result = runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Embeddings: fastembed" in result.output

    def test_ingest_invalid_embeddings_strategy_fails(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that an invalid --embeddings value is rejected by Click's Choice."""
        result = runner.invoke(ingest, [str(tmp_path), "--embeddings", "invalid"])

        assert result.exit_code == 2
        assert "Invalid value" in result.output

    @patch("zerorag.cli.ingest.store_documents")
    @patch("zerorag.cli.ingest.get_embeddings")
    @patch("zerorag.cli.ingest.split_documents")
    @patch("zerorag.cli.ingest.load_documents")
    def test_ingest_passes_embeddings_strategy(
        self, mock_load, mock_split, mock_embed, mock_store, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that --embeddings flag is passed through to get_embeddings."""
        mock_load.return_value = [Document(page_content="hello")]
        mock_split.return_value = [Document(page_content="hello")]
        mock_embed.return_value = MagicMock()

        result = runner.invoke(ingest, [str(tmp_path), "--embeddings", "openai-small"])

        assert result.exit_code == 0
        assert "Embeddings: openai-small" in result.output
        mock_embed.assert_called_once_with(strategy="openai-small")
