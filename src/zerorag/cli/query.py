from pathlib import Path

import click

from zerorag.core import get_embeddings, load_vectorstore, retrieve_documents


@click.command()
@click.argument("query")
@click.option(
    "--store",
    "store_strategy",
    default="inmemory",
    type=click.Choice(["inmemory", "chromadb"], case_sensitive=False),
    show_default=True,
    help="Vector store backend to load.",
)
@click.option(
    "--store-dir",
    default="zerorag_store",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    show_default=True,
    help="Directory where the vector store was persisted.",
)
@click.option(
    "--k",
    default=5,
    type=int,
    show_default=True,
    help="Number of top matching chunks to return.",
)
def query(query: str, store_strategy: str, store_dir: Path, k: int) -> None:
    """Query a vector store and retrieve relevant document chunks."""

    click.secho("🔍 Querying vector store...", fg="blue", bold=True)
    click.echo(f"   Store     : {store_strategy}")
    click.echo(f"   Store Dir : {store_dir.absolute()}")
    click.echo(f"   Top K     : {k}")
    click.echo()

    embeddings = get_embeddings()
    vector_store = load_vectorstore(embeddings, store_dir, strategy=store_strategy)

    results = retrieve_documents(query, vector_store, k=k)

    if not results:
        click.secho("⚠️ No results found.", fg="yellow")
        return

    for i, doc in enumerate(results, 1):
        click.secho(f"📄 Result {i}:", fg="cyan", bold=True)
        source = doc.metadata.get("source", "Unknown")
        click.echo(f"   Source: {source}")
        click.echo(f"   {doc.page_content[:200]}...")
        click.echo()

    click.secho("✅ Query complete!", fg="green", bold=True)
