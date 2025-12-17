# Research Paper RAG

A Retrieval-Augmented Generation system for querying research papers with AI-powered citations.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- **PDF Processing**: Extract text, tables, and figures from research papers
- **Intelligent Chunking**: Section-aware document chunking for better retrieval
- **Hybrid Search**: Dense (semantic) + Sparse (BM25) retrieval with reranking
- **Citation Support**: Inline citations with page and section references
- **Evaluation Dashboard**: Real-time RAG quality metrics (faithfulness, relevancy)
- **Multiple LLM Backends**: Support for Ollama (local) and OpenAI

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                        │
├─────────────────────────────────────────────────────────────┤
│                     FastAPI Backend                         │
├──────────┬──────────┬───────────────┬──────────────────────┤
│  PDF     │ Chunking │   Retrieval   │    Generation        │
│  Parser  │ Engine   │   + Rerank    │    + Citation        │
├──────────┴──────────┴───────────────┴──────────────────────┤
│                  ChromaDB Vector Store                      │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (for local LLM) or OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/research-paper-rag.git
cd research-paper-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
```

### Running with Ollama (Local LLM)

```bash
# Pull a model
ollama pull llama3.2

# Start the API server
uvicorn src.api.main:app --reload

# In another terminal, start the frontend
streamlit run frontend/app.py
```

### Running with Docker

```bash
docker-compose up --build
```

Access the application:
- Frontend: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Usage

1. **Upload Papers**: Use the sidebar to upload PDF research papers
2. **Ask Questions**: Type your question in the main input field
3. **View Results**: See the AI-generated answer with citations
4. **Check Quality**: Review evaluation metrics for the response

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a PDF paper |
| POST | `/query` | Query the papers |
| GET | `/papers` | List all papers |
| DELETE | `/papers/{id}` | Delete a paper |
| GET | `/stats` | Get system statistics |

## Project Structure

```
research-paper-rag/
├── src/
│   ├── ingestion/      # PDF parsing and chunking
│   ├── retrieval/      # Vector store and search
│   ├── generation/     # LLM and citations
│   ├── evaluation/     # RAG quality metrics
│   └── api/            # FastAPI backend
├── frontend/           # Streamlit UI
├── tests/              # Unit tests
├── data/               # Uploaded papers and vector DB
└── docker-compose.yml
```

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LLM` | `ollama` | LLM backend (`ollama` or `openai`) |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `TOP_K` | `5` | Number of chunks to retrieve |

## Tech Stack

- **PDF Processing**: PyMuPDF, pdfplumber
- **Embeddings**: sentence-transformers
- **Vector Store**: ChromaDB
- **LLM**: Ollama / OpenAI
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Evaluation**: Custom LLM-as-judge metrics

## License

MIT License
