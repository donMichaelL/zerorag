from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 32


class FastEmbedProvider:
    """Provides a LangChain Embeddings instance backed by FastEmbed (ONNX-based)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._model_name = model_name
        self._batch_size = batch_size

    def create(self) -> Embeddings:
        return FastEmbedEmbeddings(model_name=self._model_name, batch_size=self._batch_size)
