import logging
from importlib.metadata import PackageNotFoundError, version

import click

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
def cli():
    """ZeroRAG: A modular, zero-friction RAG pipeline."""


cli.add_command(zen)
