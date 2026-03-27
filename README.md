<div align="center">

# ZeroRAG

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

*A modular, production-ready Retrieval-Augmented Generation (RAG) pipeline.*

</div>

ZeroRAG is a modular, command-line Retrieval-Augmented Generation (RAG) pipeline designed for speed and simplicity. Instead of writing custom boilerplate for every new dataset, ZeroRAG allows you to transform a local folder of complex documents (PDFs, Word files) into a fully embedded vector database with a single command. Once your data is ingested, you can instantly query the database from your terminal to retrieve highly relevant text chunks—providing the perfect context window for LLM generation.


## 🚀 Install ZeroRAG

ZeroRAG is modular by design. The **core package** includes basic text processing (`.txt`) and a lightweight, file-based vector store. To keep your environment clean, you can opt-in to exactly the extra features your pipeline requires.

```bash
pip install zerorag
```
### Optional Dependencies

#### 📄 Document Parsers
Extend the core functionality to support additional file formats:

| Format | Installation Command |
| :--- | :--- |
| **PDF** | `pip install "zerorag[pdf]"` |
| **Word** | `pip install "zerorag[docx]"` |

#### 🗄️ Vector Stores
Use a different vector store backend:

| Backend | Installation Command |
| :--- | :--- |
| **ChromaDB** | `pip install "zerorag[chromadb]"` |


## ⌨️ CLI Commands

ZeroRAG provides the following commands:

### Ingest Documents

This command reads all supported documents from a source directory, splits them into manageable chunks, generates embeddings, and saves the resulting vector database.

**Syntax:**

```bash
zerorag ingest <SOURCE> [OPTIONS]
```

**Arguments:**
* `SOURCE` (Required): The folder containing your documents.

**Options:**
* `--types` *(Default: `txt`)*: Comma-separated list of file types to parse (e.g., `txt,pdf,docx`).
* `--chunk-size` *(Default: `1200`)*: Maximum number of characters per chunk.
* `--chunk-overlap` *(Default: `300`)*: Number of overlapping characters between consecutive chunks.
* `--store` *(Default: `inmemory`)*: Vector store backend (`inmemory` or `chromadb`).
* `--store-dir` *(Default: `zerorag_store`)*: Directory to persist the vector store.

*Note: Parsing non-txt files requires installing the matching optional dependencies.*

> **Embeddings:** The ingest command automatically generates embeddings using [FastEmbed](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`), a lightweight ONNX-based engine that runs locally with no API key required. The model is downloaded on first use.

**Example:**

```bash
$ zerorag ingest ./docs --types txt,pdf --store-dir ./my_store

🚀 Initializing ZeroRAG Ingestion...
   Source    : /home/user/docs
   Formats   : [txt, pdf]
   Store     : inmemory
   Store Dir : /home/user/my_store

📥 Loading documents...
✂️ Chunking documents...
🔢 Initializing embeddings...
💾 Storing vectors...
✅ Ingestion complete!
```

### Query Documents

This command runs a similarity search against a vector store and returns the most relevant document chunks.

**Syntax:**

```bash
zerorag query <QUERY> [OPTIONS]
```

**Arguments:**
* `QUERY` (Required): The search query.

**Options:**
* `--store` *(Default: `inmemory`)*: Vector store backend to load (`inmemory` or `chromadb`).
* `--store-dir` *(Default: `zerorag_store`)*: Directory where the vector store was persisted during ingestion.
* `--k` *(Default: `5`)*: Number of top matching chunks to return.

**Example:**

```bash
$ zerorag query "What are the main benefits of RAG?" --store-dir ./my_store --k 2

🔍 Querying vector store...
   Store     : inmemory
   Store Dir : /home/user/my_store
   Top K     : 2

📄 Result 1:
   Source: intro.pdf | Page: 2
   RAG combines retrieval with generation to ground LLM responses in
   real data, reducing hallucinations and improving factual accuracy...

📄 Result 2:
   Source: overview.txt
   The primary benefits include reduced hallucination, up-to-date answers
   from private data, and full traceability back to source documents...

✅ Query complete!
```

## 🐛 Reporting Bugs & Feature Requests

We are constantly looking to improve ZeroRAG. If you encounter a bug or have an idea for a new feature (like a new vector store or document loader), please let us know!

To report a bug or submit a feature request, visit the [Issues page](https://github.com/donMichaelL/zerorag/issues).
