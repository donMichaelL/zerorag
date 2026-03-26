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
