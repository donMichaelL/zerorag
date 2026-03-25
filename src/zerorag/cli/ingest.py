from pathlib import Path

import click

from zerorag.core import load_documents, split_documents


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
@click.option(
    "--chunk-size",
    default=1200,
    type=int,
    show_default=True,
    help="Maximum number of characters per chunk.",
)
@click.option(
    "--chunk-overlap",
    default=300,
    type=int,
    show_default=True,
    help="Number of overlapping characters between consecutive chunks.",
)
def ingest(source: Path, types: list[str], chunk_size: int, chunk_overlap: int) -> None:
    """Ingest documents from a folder and build a vector store."""

    click.secho("🚀 Initializing ZeroRAG Ingestion...", fg="blue", bold=True)
    click.echo(f"   Source    : {source.absolute()}")
    click.echo(f"   Formats   : [{', '.join(types)}]")
    click.echo()

    click.echo("📥 Loading documents...")
    documents = load_documents(source, types)

    if not documents:
        click.secho("⚠️ No pages found. Nothing to split.", fg="yellow")
        return

    click.echo("✂️  Chunking documents...")
    _chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    click.secho("✅ Ingestion complete!", fg="green", bold=True)
