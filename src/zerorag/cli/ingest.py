from pathlib import Path

import click

from zerorag.core.loaders import load_documents


def parse_comma_separated_types(ctx: click.Context, param: click.Parameter, value: str | None) -> list[str]:
    """Intercept the CLI string, sanitize it, and return a clean list."""
    if not value:
        return ["txt"]
    return [t.strip().lower() for t in value.split(",")]


@click.command()
@click.argument("source", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--types",
    default="txt",
    callback=parse_comma_separated_types,
    help="Comma-separated file types (e.g., pdf, docx, txt).",
)
def ingest(source: Path, types: list[str]) -> None:
    """Ingest documents from a folder and build a vector store."""

    click.secho("🚀 Initializing ZeroRAG Ingestion...", fg="blue", bold=True)
    click.echo(f"📂 Source Directory : {source.absolute()}")
    click.echo(f"📄 Target Formats   : [{', '.join(types)}]")
    click.echo("\n... Scanning and loading documents ...")

    load_documents(source, types)
