from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from langchain_core.documents import Document

from zerorag.cli.ask import ask


class TestAskCommand:
    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a standard Click CLI runner for the test class."""
        return CliRunner()

    def test_ask_missing_argument_fails(self, runner: CliRunner) -> None:
        """Test that omitting the required question argument throws an error."""
        result = runner.invoke(ask)

        assert result.exit_code == 2
        assert "Missing argument 'QUESTION'" in result.output

    def test_ask_nonexistent_store_dir_fails(self, runner: CliRunner) -> None:
        """Test that providing a path that doesn't exist triggers Click's validation."""
        result = runner.invoke(ask, ["some question", "--store-dir", "./ghost_folder"])

        assert result.exit_code == 2
        assert "Directory './ghost_folder' does not exist" in result.output

    def test_ask_invalid_store_strategy_fails(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that an invalid --store value is rejected by Click's Choice."""
        result = runner.invoke(ask, ["some question", "--store-dir", str(tmp_path), "--store", "invalid"])

        assert result.exit_code == 2
        assert "Invalid value" in result.output

    def test_ask_invalid_llm_strategy_fails(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that an invalid --llm value is rejected by Click's Choice."""
        result = runner.invoke(ask, ["some question", "--store-dir", str(tmp_path), "--llm", "invalid"])

        assert result.exit_code == 2
        assert "Invalid value" in result.output

    def test_ask_invalid_embeddings_strategy_fails(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that an invalid --embeddings value is rejected by Click's Choice."""
        result = runner.invoke(ask, ["some question", "--store-dir", str(tmp_path), "--embeddings", "invalid"])

        assert result.exit_code == 2
        assert "Invalid value" in result.output

    def test_ask_default_options(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that default options are displayed correctly."""
        with (
            patch("zerorag.cli.ask.get_embeddings") as mock_embed,
            patch("zerorag.cli.ask.load_vectorstore") as mock_load,
            patch("zerorag.cli.ask.retrieve_documents", return_value=[]),
        ):
            mock_embed.return_value = MagicMock()
            mock_load.return_value = MagicMock()

            result = runner.invoke(ask, ["some question", "--store-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Question  : some question" in result.output
        assert "LLM       : openai-mini" in result.output
        assert "Embeddings: fastembed" in result.output
        assert "Store     : inmemory" in result.output
        assert "Top K     : 5" in result.output

    def test_ask_no_results_shows_warning(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that an empty result set shows a warning and exits early."""
        with (
            patch("zerorag.cli.ask.get_embeddings") as mock_embed,
            patch("zerorag.cli.ask.load_vectorstore") as mock_load,
            patch("zerorag.cli.ask.retrieve_documents", return_value=[]),
        ):
            mock_embed.return_value = MagicMock()
            mock_load.return_value = MagicMock()

            result = runner.invoke(ask, ["some question", "--store-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("zerorag.cli.ask.generate_answer")
    @patch("zerorag.cli.ask.get_llm")
    @patch("zerorag.cli.ask.retrieve_documents")
    @patch("zerorag.cli.ask.load_vectorstore")
    @patch("zerorag.cli.ask.get_embeddings")
    def test_ask_full_pipeline(
        self, mock_embed, mock_load, mock_retrieve, mock_llm, mock_generate, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test the full pipeline with mocked core functions."""
        mock_embed.return_value = MagicMock()
        mock_load.return_value = MagicMock()
        mock_retrieve.return_value = [Document(page_content="context chunk", metadata={"source": "doc.txt"})]
        mock_llm.return_value = MagicMock()
        mock_generate.return_value = "The answer is 42."

        result = runner.invoke(ask, ["some question", "--store-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Retrieving relevant chunks" in result.output
        assert "Generating answer" in result.output
        assert "The answer is 42." in result.output
        assert "Done!" in result.output
        mock_embed.assert_called_once()
        mock_load.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_llm.assert_called_once()
        mock_generate.assert_called_once()

    @patch("zerorag.cli.ask.generate_answer")
    @patch("zerorag.cli.ask.get_llm")
    @patch("zerorag.cli.ask.retrieve_documents")
    @patch("zerorag.cli.ask.load_vectorstore")
    @patch("zerorag.cli.ask.get_embeddings")
    def test_ask_passes_llm_strategy(
        self, mock_embed, mock_load, mock_retrieve, mock_llm, mock_generate, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that --llm flag is passed through to get_llm."""
        mock_embed.return_value = MagicMock()
        mock_load.return_value = MagicMock()
        mock_retrieve.return_value = [Document(page_content="chunk", metadata={"source": "doc.txt"})]
        mock_llm.return_value = MagicMock()
        mock_generate.return_value = "answer"

        result = runner.invoke(ask, ["some question", "--store-dir", str(tmp_path), "--llm", "openai"])

        assert result.exit_code == 0
        assert "LLM       : openai" in result.output
        mock_llm.assert_called_once_with(strategy="openai")

    @patch("zerorag.cli.ask.generate_answer")
    @patch("zerorag.cli.ask.get_llm")
    @patch("zerorag.cli.ask.retrieve_documents")
    @patch("zerorag.cli.ask.load_vectorstore")
    @patch("zerorag.cli.ask.get_embeddings")
    def test_ask_passes_embeddings_strategy(
        self, mock_embed, mock_load, mock_retrieve, mock_llm, mock_generate, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that --embeddings flag is passed through to get_embeddings."""
        mock_embed.return_value = MagicMock()
        mock_load.return_value = MagicMock()
        mock_retrieve.return_value = [Document(page_content="chunk", metadata={"source": "doc.txt"})]
        mock_llm.return_value = MagicMock()
        mock_generate.return_value = "answer"

        result = runner.invoke(ask, ["some question", "--store-dir", str(tmp_path), "--embeddings", "openai-large"])

        assert result.exit_code == 0
        mock_embed.assert_called_once_with(strategy="openai-large")

    @patch("zerorag.cli.ask.generate_answer")
    @patch("zerorag.cli.ask.get_llm")
    @patch("zerorag.cli.ask.retrieve_documents")
    @patch("zerorag.cli.ask.load_vectorstore")
    @patch("zerorag.cli.ask.get_embeddings")
    def test_ask_passes_store_strategy(
        self, mock_embed, mock_load, mock_retrieve, mock_llm, mock_generate, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that --store flag is passed through to load_vectorstore."""
        mock_embed.return_value = MagicMock()
        mock_load.return_value = MagicMock()
        mock_retrieve.return_value = [Document(page_content="chunk", metadata={"source": "doc.txt"})]
        mock_llm.return_value = MagicMock()
        mock_generate.return_value = "answer"

        result = runner.invoke(ask, ["some question", "--store-dir", str(tmp_path), "--store", "chromadb"])

        assert result.exit_code == 0
        assert "Store     : chromadb" in result.output
        mock_load.assert_called_once()
        assert mock_load.call_args[1]["strategy"] == "chromadb"
