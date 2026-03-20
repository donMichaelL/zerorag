from pathlib import Path

from langchain_core.documents import Document


class MockLoaderStrategy:
    def load(self, source: Path) -> list[Document]:
        return [
            Document(
                page_content="mock data",
                metadata={"source": str(source)},
            )
        ]


def test_document_loader_strategy_contract():
    """
    Test that a custom class implementing the DocumentLoaderStrategy
    protocol successfully fulfills the required contract.
    """
    strategy = MockLoaderStrategy()
    test_path = Path("/fake/directory/path")
    documents = strategy.load(test_path)

    assert isinstance(documents, list), "Should return a list"
    assert len(documents) == 1, "Should contain exactly one document"
