import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

from .base import LLMStrategy
from .openai import LLM_MODELS, OpenAILLMProvider
from .prompts import DEFAULT_RAG_TEMPLATE

logger = logging.getLogger(__name__)


LLM_REGISTRY: dict[str, tuple[type[LLMStrategy], dict]] = {
    "openai-mini": (OpenAILLMProvider, {"model_name": LLM_MODELS["openai-mini"]}),
    "openai": (OpenAILLMProvider, {"model_name": LLM_MODELS["openai"]}),
}


def get_llm(strategy: str = "openai-mini") -> BaseLanguageModel:
    """
    Create and return a LangChain LLM instance.

    Args:
        strategy: The LLM provider to use (must be a key in LLM_REGISTRY).

    Returns:
        A LangChain BaseLanguageModel instance.
    """
    entry = LLM_REGISTRY.get(strategy)
    if not entry:
        logger.warning(f"Unknown LLM strategy '{strategy}', falling back to 'openai-mini'")
        entry = LLM_REGISTRY["openai-mini"]

    provider_cls, kwargs = entry
    return provider_cls(**kwargs).create()


def generate_answer(question: str, documents: list[Document], llm: BaseLanguageModel) -> str:
    """
    Generate an answer using retrieved documents as context.

    Args:
        question: The user's question.
        documents: Retrieved document chunks to use as context.
        llm: A LangChain LLM instance.

    Returns:
        The generated answer string.
    """
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = ChatPromptTemplate.from_template(DEFAULT_RAG_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context})
    return response.content


__all__ = ["generate_answer", "get_llm"]
