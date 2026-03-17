import click


@click.command(hidden=True)
def zen():
    """A secret command for those who seek the path of ZeroRAG."""
    click.echo("")
    click.secho("🧘 ZeroRAG Zen", fg="cyan", bold=True)
    click.echo("-" * 20)
    click.echo("1. Simple is better than complex.")
    click.echo("2. Data is heavy; retrieval should be light.")
    click.echo("3. Don't give up; the situation is serious.")
    click.echo("4. Precision beats volume.")
    click.echo("-" * 20)
    click.echo("")
