from langchain_core.documents import Document

from zerorag.core.splitters import split_documents


class TestSplitDocumentsEngine:
    """Test suite for the main registry-based splitting engine."""

    def test_splits_documents_with_default_strategy(self):
        """Test that the engine successfully splits documents using the default strategy."""
        content = "Word " * 500
        documents = [Document(page_content=content)]
        chunks = split_documents(documents)

        assert len(chunks) > 1

    def test_unknown_strategy_falls_back_to_recursive(self):
        """Test that an unknown strategy name falls back to the recursive splitter."""
        content = "Word " * 500
        documents = [Document(page_content=content)]
        chunks = split_documents(documents, strategy="nonexistent")

        assert len(chunks) > 1

    def test_empty_documents_returns_empty(self):
        """Test that splitting an empty list returns an empty list."""
        chunks = split_documents([])

        assert len(chunks) == 0
