from click.testing import CliRunner

from zerorag.cli import cli


def test_version_flag_outputs_correct_string():
    """Test that the --version flag outputs the correct version string."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "ZeroRAG, version" in result.output


def test_main_execution_without_args():
    """Test that running the CLI without a subcommand correctly shows the help/usage."""
    runner = CliRunner()
    result = runner.invoke(cli)

    assert result.exit_code == 2
    assert "ZeroRAG: A modular, zero-friction RAG pipeline." in result.output
