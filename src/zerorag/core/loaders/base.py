from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document


class DocumentLoaderStrategy(Protocol):
    """Protocol defining the standard interface for all file loaders."""

    def load(self, source: Path) -> list[Document]: ...
