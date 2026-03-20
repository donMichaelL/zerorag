from pathlib import Path

import pytest
from click.testing import CliRunner

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

    def test_ingest_valid_directory_succeeds(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that a valid directory executes the command successfully."""
        result = runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Initializing ZeroRAG Ingestion" in result.output

    def test_ingest_default_types(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that omitting --types correctly prints the default."""
        result = runner.invoke(ingest, [str(tmp_path)])

        assert result.exit_code == 0
        assert "Target Formats   : [txt]" in result.output

    def test_ingest_custom_types(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test passing custom types with messy spacing and casing."""
        result = runner.invoke(ingest, [str(tmp_path), "--types", "TXT, pdf,  Csv "])

        assert "Target Formats   : [txt, pdf, csv]" in result.output
