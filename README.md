# rag-brain

A flexible RAG (Retrieval-Augmented Generation) system manager for processing documents, chunking text, and managing a ChromaDB knowledge base.

## Features

- **Document Ingestion**: Recursively process folders and ingest files (PDF, Markdown, Python, Text).
- **Flexible Chunking**: Supports multiple chunking strategies including:
  - Recursive
  - Markdown
  - Semantic
  - Character
  - Endline
- **ChromaDB Integration**: Built-in support for vector storage and retrieval.
- **RAG Capabilities**: Query your knowledge base with the built-in CLI or MCP server.

## Installation

Prerequisites:
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Python 3.12+

```bash
# Clone the repository
git clone https://github.com/cleanunicorn/process.git
cd process

# Install dependencies using Makefile
make install
```

## Usage

### 1. Start the Database
First, you need to start the ChromaDB server:

```bash
make run-db
```
This will start ChromaDB on `localhost:8000`.

### 2. Ingest Documents
Load your documents into the knowledge base:

```bash
uv run python main.py rag ./your-docs-folder my-collection-name \
  --strategy recursive \
  --chunk-size 1000 \
  --chunk-overlap 200
```

### 3. Query the Knowledge Base
Search for content:

```bash
uv run python main.py query "your search query" my-collection-name
```

### 4. Test Chunking (Optional)
Test how a specific file will be chunked before ingesting:

```bash
uv run python main.py chunk ./path/to/file.md --strategy markdown --chunk-size 500
```

## Development

- **Linting**:
  ```bash
  make lint
  ```

- **Run Tests** (Manual verification):
  ```bash
  make run
  ```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
