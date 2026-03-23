from pathlib import Path

import pytest
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document


@pytest.fixture
def fast_directory_loader(monkeypatch) -> None:
    """
    Intercept LangChain's DirectoryLoader to completely bypass the ThreadPoolExecutor.
    This drops the execution time from 8s to ~0.01s while still perfectly validating
    our file routing and nested folder logic.
    """

    def bypass_threadpool_load(self):
        docs = []
        for file_path in Path(self.path).glob(self.glob):
            if file_path.is_file():
                docs.append(Document(page_content="speed test", metadata={"source": str(file_path)}))
        return docs

    monkeypatch.setattr(DirectoryLoader, "load", bypass_threadpool_load)
