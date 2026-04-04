from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings

from zerorag.core.embeddings.openai import OpenAIEmbedProvider
from zerorag.exceptions import MissingDependencyError


class TestOpenAIEmbedProvider:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("langchain_openai.OpenAIEmbeddings")
    def test_create_returns_embeddings_instance(self, mock_cls):
        """Test that create() returns a LangChain Embeddings instance."""
        mock_instance = MagicMock(spec=Embeddings)
        mock_cls.return_value = mock_instance

        provider = OpenAIEmbedProvider()
        embeddings = provider.create()

        assert embeddings is mock_instance
        mock_cls.assert_called_once_with(model="text-embedding-3-small")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("langchain_openai.OpenAIEmbeddings")
    def test_create_passes_custom_model_name(self, mock_cls):
        """Test that a custom model name is forwarded to OpenAIEmbeddings."""
        provider = OpenAIEmbedProvider(model_name="text-embedding-3-large")
        provider.create()

        mock_cls.assert_called_once_with(model="text-embedding-3-large")

    @patch("zerorag.core.embeddings.openai.importlib.util.find_spec", return_value=None)
    def test_create_raises_when_package_missing(self, mock_spec):
        """Test that a missing langchain-openai package raises MissingDependencyError."""
        provider = OpenAIEmbedProvider()

        with pytest.raises(MissingDependencyError, match="pip install"):
            provider.create()

    @patch.dict("os.environ", {}, clear=True)
    def test_create_raises_when_api_key_missing(self):
        """Test that a missing OPENAI_API_KEY raises MissingDependencyError."""
        provider = OpenAIEmbedProvider()

        with pytest.raises(MissingDependencyError, match="OPENAI_API_KEY"):
            provider.create()
