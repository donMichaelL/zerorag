from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings

from zerorag.core.embeddings import get_embeddings


@patch("zerorag.core.embeddings.fastembed.FastEmbedEmbeddings")
class TestGetEmbeddingsEngine:
    """Test suite for the main registry-based embeddings engine."""

    def test_returns_embeddings_with_default_strategy(self, mock_cls):
        """Test that the engine returns an Embeddings instance using the default strategy."""
        mock_cls.return_value = MagicMock(spec=Embeddings)
        embeddings = get_embeddings()

        assert isinstance(embeddings, Embeddings)

    def test_unknown_strategy_falls_back_to_fastembed(self, mock_cls):
        """Test that an unknown strategy name falls back to fastembed."""
        mock_cls.return_value = MagicMock(spec=Embeddings)
        embeddings = get_embeddings(strategy="nonexistent")

        assert isinstance(embeddings, Embeddings)

    def test_explicit_fastembed_strategy(self, mock_cls):
        """Test that explicitly requesting fastembed works."""
        mock_cls.return_value = MagicMock(spec=Embeddings)
        embeddings = get_embeddings(strategy="fastembed")

        assert isinstance(embeddings, Embeddings)


@patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
@patch("langchain_openai.OpenAIEmbeddings")
class TestGetEmbeddingsOpenAI:
    """Test suite for OpenAI strategies in the registry."""

    def test_openai_small_strategy(self, mock_cls):
        """Test that openai-small strategy returns an Embeddings instance."""
        mock_cls.return_value = MagicMock(spec=Embeddings)
        embeddings = get_embeddings(strategy="openai-small")

        assert isinstance(embeddings, Embeddings)
        mock_cls.assert_called_once_with(model="text-embedding-3-small")

    def test_openai_large_strategy(self, mock_cls):
        """Test that openai-large strategy returns an Embeddings instance."""
        mock_cls.return_value = MagicMock(spec=Embeddings)
        embeddings = get_embeddings(strategy="openai-large")

        assert isinstance(embeddings, Embeddings)
        mock_cls.assert_called_once_with(model="text-embedding-3-large")
