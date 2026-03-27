from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from langchain_core.documents import Document

from zerorag.cli.query import query


class TestQueryCommand:
    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a standard Click CLI runner for the test class."""
        return CliRunner()

    def test_query_missing_argument_fails(self, runner: CliRunner) -> None:
        """Test that omitting the required query argument throws an error."""
        result = runner.invoke(query)

        assert result.exit_code == 2
        assert "Missing argument 'QUERY'" in result.output

    def test_query_nonexistent_store_dir_fails(self, runner: CliRunner) -> None:
        """Test that providing a path that doesn't exist triggers Click's validation."""
        result = runner.invoke(query, ["some query", "--store-dir", "./ghost_folder"])

        assert result.exit_code == 2
        assert "Directory './ghost_folder' does not exist" in result.output

    def test_query_invalid_store_strategy_fails(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that an invalid --store value is rejected by Click's Choice."""
        result = runner.invoke(query, ["some query", "--store-dir", str(tmp_path), "--store", "invalid"])

        assert result.exit_code == 2
        assert "Invalid value" in result.output

    def test_query_default_options(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that default store and k options are displayed correctly."""
        with (
            patch("zerorag.cli.query.get_embeddings") as mock_embed,
            patch("zerorag.cli.query.load_vectorstore") as mock_load,
            patch("zerorag.cli.query.retrieve_documents", return_value=[]),
        ):
            mock_embed.return_value = MagicMock()
            mock_load.return_value = MagicMock()

            result = runner.invoke(query, ["some query", "--store-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Store     : inmemory" in result.output
        assert "Top K     : 5" in result.output

    def test_query_no_results_shows_warning(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that an empty result set shows a warning and exits early."""
        with (
            patch("zerorag.cli.query.get_embeddings") as mock_embed,
            patch("zerorag.cli.query.load_vectorstore") as mock_load,
            patch("zerorag.cli.query.retrieve_documents", return_value=[]),
        ):
            mock_embed.return_value = MagicMock()
            mock_load.return_value = MagicMock()

            result = runner.invoke(query, ["some query", "--store-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("zerorag.cli.query.retrieve_documents")
    @patch("zerorag.cli.query.load_vectorstore")
    @patch("zerorag.cli.query.get_embeddings")
    def test_query_displays_results(
        self, mock_embed, mock_load, mock_retrieve, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that retrieved documents are displayed with source and content."""
        mock_embed.return_value = MagicMock()
        mock_load.return_value = MagicMock()
        mock_retrieve.return_value = [
            Document(page_content="First result content", metadata={"source": "doc1.pdf"}),
            Document(page_content="Second result content", metadata={"source": "doc2.txt"}),
        ]

        result = runner.invoke(query, ["some query", "--store-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Result 1:" in result.output
        assert "Source: doc1.pdf" in result.output
        assert "First result content" in result.output
        assert "Result 2:" in result.output
        assert "Source: doc2.txt" in result.output
        assert "Query complete" in result.output

    @patch("zerorag.cli.query.retrieve_documents")
    @patch("zerorag.cli.query.load_vectorstore")
    @patch("zerorag.cli.query.get_embeddings")
    def test_query_passes_store_strategy(
        self, mock_embed, mock_load, mock_retrieve, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that --store flag is passed through to load_vectorstore."""
        mock_embed.return_value = MagicMock()
        mock_load.return_value = MagicMock()
        mock_retrieve.return_value = []

        result = runner.invoke(query, ["some query", "--store-dir", str(tmp_path), "--store", "chromadb"])

        assert result.exit_code == 0
        assert "Store     : chromadb" in result.output
        mock_load.assert_called_once()
        assert mock_load.call_args[1]["strategy"] == "chromadb"
