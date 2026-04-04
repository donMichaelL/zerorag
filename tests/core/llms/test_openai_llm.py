from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseLanguageModel

from zerorag.core.llms.openai import OpenAILLMProvider
from zerorag.exceptions import MissingDependencyError


class TestOpenAILLMProvider:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("langchain_openai.ChatOpenAI")
    def test_create_returns_llm_instance(self, mock_cls):
        """Test that create() returns a LangChain BaseLanguageModel instance."""
        mock_instance = MagicMock(spec=BaseLanguageModel)
        mock_cls.return_value = mock_instance

        provider = OpenAILLMProvider()
        llm = provider.create()

        assert llm is mock_instance
        mock_cls.assert_called_once_with(model="gpt-4.1-mini")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("langchain_openai.ChatOpenAI")
    def test_create_passes_custom_model_name(self, mock_cls):
        """Test that a custom model name is forwarded to ChatOpenAI."""
        provider = OpenAILLMProvider(model_name="gpt-4.1")
        provider.create()

        mock_cls.assert_called_once_with(model="gpt-4.1")

    @patch("zerorag.core.llms.openai.importlib.util.find_spec", return_value=None)
    def test_create_raises_when_package_missing(self, mock_spec):
        """Test that a missing langchain-openai package raises MissingDependencyError."""
        provider = OpenAILLMProvider()

        with pytest.raises(MissingDependencyError, match="pip install"):
            provider.create()

    @patch.dict("os.environ", {}, clear=True)
    def test_create_raises_when_api_key_missing(self):
        """Test that a missing OPENAI_API_KEY raises MissingDependencyError."""
        provider = OpenAILLMProvider()

        with pytest.raises(MissingDependencyError, match="OPENAI_API_KEY"):
            provider.create()
