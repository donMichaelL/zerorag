from pathlib import Path

import click

from zerorag.core import get_embeddings, load_documents, split_documents, store_documents


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
@click.option(
    "--store",
    "store_strategy",
    default="inmemory",
    type=click.Choice(["inmemory", "chromadb"], case_sensitive=False),
    show_default=True,
    help="Vector store backend to use.",
)
@click.option(
    "--store-dir",
    default="zerorag_store",
    type=click.Path(path_type=Path),
    show_default=True,
    help="Directory to persist the vector store.",
)
def ingest(
    source: Path, types: list[str], chunk_size: int, chunk_overlap: int, store_strategy: str, store_dir: Path
) -> None:
    """Ingest documents from a folder and build a vector store."""

    click.secho("🚀 Initializing ZeroRAG Ingestion...", fg="blue", bold=True)
    click.echo(f"   Source    : {source.absolute()}")
    click.echo(f"   Formats   : [{', '.join(types)}]")
    click.echo(f"   Store     : {store_strategy}")
    click.echo(f"   Store Dir : {store_dir.absolute()}")
    click.echo()

    click.echo("📥 Loading documents...")
    documents = load_documents(source, types)

    if not documents:
        click.secho("⚠️ No pages found. Nothing to split.", fg="yellow")
        return

    click.echo("✂️ Chunking documents...")
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    click.echo("🔢 Initializing embeddings...")
    embeddings = get_embeddings()

    click.echo("💾 Storing vectors...")
    store_documents(chunks, embeddings, store_dir, strategy=store_strategy)

    click.secho("✅ Ingestion complete!", fg="green", bold=True)
