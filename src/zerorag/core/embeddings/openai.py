import importlib.util
import os

from langchain_core.embeddings import Embeddings

from zerorag.exceptions import MissingDependencyError


class OpenAIEmbedProvider:
    """Provides a LangChain Embeddings instance backed by OpenAI."""

    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        self._model_name = model_name

    def create(self) -> Embeddings:
        if importlib.util.find_spec("langchain_openai") is None:
            raise MissingDependencyError("OpenAI support is missing. Run: pip install 'zerorag[openai]'")

        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingDependencyError(
                "OPENAI_API_KEY environment variable is required when using OpenAI embeddings."
            )

        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=self._model_name)
