from click.testing import CliRunner

from zerorag.cli.zen import zen


def test_zen_command_output():
    """Test that the secret zen command outputs the expected philosophy."""
    runner = CliRunner()
    result = runner.invoke(zen)

    assert result.exit_code == 0
    assert "ZeroRAG Zen" in result.output
    assert "Simple is better than complex." in result.output
    assert "Don't give up; the situation is serious." in result.output
