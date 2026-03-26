from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 3 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 3


class MockEmbeddingsStrategy:
    def create(self) -> Embeddings:
        return FakeEmbeddings()


def test_embeddings_strategy_contract():
    """
    Test that a custom class implementing the EmbeddingsStrategy
    protocol successfully fulfills the required contract.
    """
    strategy = MockEmbeddingsStrategy()
    embeddings = strategy.create()

    assert isinstance(embeddings, Embeddings), "Should return a LangChain Embeddings instance"
