from typing import Protocol

from langchain_core.language_models import BaseLanguageModel


class LLMStrategy(Protocol):
    """Protocol defining the standard interface for all LLM providers."""

    def create(self) -> BaseLanguageModel: ...
