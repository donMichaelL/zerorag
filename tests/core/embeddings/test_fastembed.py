from unittest.mock import MagicMock, patch

from langchain_core.embeddings import Embeddings

from zerorag.core.embeddings.fastembed import FastEmbedProvider


class TestFastEmbedProvider:
    @patch("zerorag.core.embeddings.fastembed.FastEmbedEmbeddings")
    def test_create_returns_embeddings_instance(self, mock_cls):
        """Test that create() returns a LangChain Embeddings instance."""
        mock_instance = MagicMock(spec=Embeddings)
        mock_cls.return_value = mock_instance

        provider = FastEmbedProvider()
        embeddings = provider.create()

        assert embeddings is mock_instance
        mock_cls.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5", batch_size=32)

    @patch("zerorag.core.embeddings.fastembed.FastEmbedEmbeddings")
    def test_create_passes_custom_model_name(self, mock_cls):
        """Test that a custom model name is forwarded to FastEmbedEmbeddings."""
        provider = FastEmbedProvider(model_name="custom/model")
        provider.create()

        mock_cls.assert_called_once_with(model_name="custom/model", batch_size=32)
