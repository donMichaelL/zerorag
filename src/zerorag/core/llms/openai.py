import importlib.util
import os

from langchain_core.language_models import BaseLanguageModel

from zerorag.exceptions import MissingDependencyError

LLM_MODELS: dict[str, str] = {
    "openai-mini": "gpt-4.1-mini",
    "openai": "gpt-4.1",
}


class OpenAILLMProvider:
    """Provides a LangChain LLM instance backed by OpenAI."""

    def __init__(self, model_name: str = "gpt-4.1-mini") -> None:
        self._model_name = model_name

    def create(self) -> BaseLanguageModel:
        if importlib.util.find_spec("langchain_openai") is None:
            raise MissingDependencyError("OpenAI support is missing. Run: pip install 'zerorag[openai]'")

        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingDependencyError("OPENAI_API_KEY environment variable is required when using OpenAI LLMs.")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=self._model_name)
