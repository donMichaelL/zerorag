import pytest
from langchain_core.documents import Document

from zerorag.core.splitters.recursive import RecursiveCharacterSplitter


@pytest.fixture
def splitter() -> RecursiveCharacterSplitter:
    return RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)


@pytest.fixture
def single_short_document() -> list[Document]:
    return [Document(page_content="Short text.", metadata={"source": "test.txt"})]


@pytest.fixture
def single_long_document() -> list[Document]:
    content = "This is a sentence that repeats. " * 20
    return [Document(page_content=content, metadata={"source": "long.txt"})]


class TestRecursiveCharacterSplitter:
    def test_short_document_stays_single_chunk(
        self, splitter: RecursiveCharacterSplitter, single_short_document: list[Document]
    ):
        """Test that a document smaller than chunk_size is not split."""
        chunks = splitter.split(single_short_document)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_long_document_is_split_into_multiple_chunks(
        self, splitter: RecursiveCharacterSplitter, single_long_document: list[Document]
    ):
        """Test that a document larger than chunk_size is split into multiple chunks."""
        chunks = splitter.split(single_long_document)

        assert len(chunks) > 1

    def test_chunks_respect_max_size(self, splitter: RecursiveCharacterSplitter, single_long_document: list[Document]):
        """Test that no chunk exceeds the configured chunk_size."""
        chunks = splitter.split(single_long_document)

        for chunk in chunks:
            assert len(chunk.page_content) <= 50

    def test_chunks_preserve_metadata(
        self, splitter: RecursiveCharacterSplitter, single_long_document: list[Document]
    ):
        """Test that source metadata is preserved on every chunk."""
        chunks = splitter.split(single_long_document)

        for chunk in chunks:
            assert chunk.metadata["source"] == "long.txt"

    def test_chunks_include_start_index(
        self, splitter: RecursiveCharacterSplitter, single_long_document: list[Document]
    ):
        """Test that add_start_index=True populates start_index in metadata."""
        chunks = splitter.split(single_long_document)

        for chunk in chunks:
            assert "start_index" in chunk.metadata

    def test_empty_document_list(self, splitter: RecursiveCharacterSplitter):
        """Test that splitting an empty list returns an empty list."""
        chunks = splitter.split([])

        assert isinstance(chunks, list)
        assert len(chunks) == 0
