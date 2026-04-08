# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-08

### Added
- **CLI commands:** `ingest`, `query`, and `ask` for the full RAG pipeline.
- **Document loaders:** Support for `.txt`, `.pdf`, and `.docx` files with multithreaded loading.
- **Text splitting:** Recursive character text splitter with configurable chunk size and overlap.
- **Embedding providers:** FastEmbed (local, free) as default, with OpenAI (`text-embedding-3-small`, `text-embedding-3-large`) as optional providers.
- **Vector store backends:** InMemory (file-based) and ChromaDB.
- **LLM integration:** OpenAI (`openai-mini`, `openai`) for the `ask` command with a built-in RAG prompt template.
- **Retrieval:** Similarity search with configurable top-k results.

## [0.0.1] - 2026-03-18

### Added
- Initial package release to reserve the name on PyPI.
