from pathlib import Path

import click

from zerorag.core import generate_answer, get_embeddings, get_llm, load_vectorstore, retrieve_documents


@click.command()
@click.argument("question")
@click.option(
    "--llm",
    "llm_strategy",
    default="openai-mini",
    type=click.Choice(["openai-mini", "openai"], case_sensitive=False),
    show_default=True,
    help="LLM to use for generation.",
)
@click.option(
    "--embeddings",
    "embeddings_strategy",
    default="fastembed",
    type=click.Choice(["fastembed", "openai-small", "openai-large"], case_sensitive=False),
    show_default=True,
    help="Embedding provider to use. Must match the provider used during ingestion.",
)
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
    help="Number of top matching chunks to retrieve as context.",
)
def ask(
    question: str,
    llm_strategy: str,
    embeddings_strategy: str,
    store_strategy: str,
    store_dir: Path,
    k: int,
) -> None:
    """Ask a question and get an LLM-generated answer grounded in your documents."""

    click.secho("🤖 Asking LLM...", fg="blue", bold=True)
    click.echo(f"   Question  : {question}")
    click.echo(f"   LLM       : {llm_strategy}")
    click.echo(f"   Embeddings: {embeddings_strategy}")
    click.echo(f"   Store     : {store_strategy}")
    click.echo(f"   Store Dir : {store_dir.absolute()}")
    click.echo(f"   Top K     : {k}")
    click.echo()

    click.echo("🔍 Retrieving relevant chunks...")
    embeddings = get_embeddings(strategy=embeddings_strategy)
    vector_store = load_vectorstore(embeddings, store_dir, strategy=store_strategy)
    results = retrieve_documents(question, vector_store, k=k)

    if not results:
        click.secho("⚠️ No results found. Cannot generate an answer without context.", fg="yellow")
        return

    click.echo("💬 Generating answer...")
    llm = get_llm(strategy=llm_strategy)
    answer = generate_answer(question, results, llm)

    click.echo()
    click.secho("📝 Answer:", fg="cyan", bold=True)
    click.echo(f"   {answer}")
    click.echo()

    click.secho("✅ Done!", fg="green", bold=True)
