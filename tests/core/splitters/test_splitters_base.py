from langchain_core.documents import Document


class MockSplitterStrategy:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 300) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Document]:  # noqa: ARG002
        return [
            Document(
                page_content="mock chunk",
                metadata={"source": "mock"},
            )
        ]


def test_document_splitter_strategy_contract():
    """
    Test that a custom class implementing the DocumentSplitterStrategy
    protocol successfully fulfills the required contract.
    """
    strategy = MockSplitterStrategy()
    documents = [Document(page_content="some content")]
    chunks = strategy.split(documents)

    assert isinstance(chunks, list), "Should return a list"
    assert len(chunks) == 1, "Should contain exactly one chunk"
