import logging
from importlib.metadata import PackageNotFoundError, version

import click

from .ask import ask
from .ingest import ingest
from .query import query
from .zen import zen

logger = logging.getLogger(__name__)


def get_version() -> str:
    """Dynamically retrieve the version of zerorag."""
    try:
        return version("zerorag")
    except PackageNotFoundError:
        return "0.1.0-dev"


@click.group()
@click.version_option(version=get_version(), prog_name="ZeroRAG")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose debug logging.")
def cli(verbose: bool) -> None:
    """ZeroRAG: A modular, zero-friction RAG pipeline."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s", force=True)
    logger.debug("Logging system initialized in CLI.")


cli.add_command(zen)
cli.add_command(ingest)
cli.add_command(query)
cli.add_command(ask)
