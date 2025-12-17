# Technical Documentation

This document explains the theoretical foundations, architectural decisions, and implementation details behind the Research Paper RAG system.

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [System Architecture](#system-architecture)
3. [PDF Processing Pipeline](#pdf-processing-pipeline)
4. [Chunking Strategies](#chunking-strategies)
5. [Embedding Models](#embedding-models)
6. [Vector Databases](#vector-databases)
7. [Retrieval Strategies](#retrieval-strategies)
8. [Reranking](#reranking)
9. [Generation with Citations](#generation-with-citations)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Design Decisions](#design-decisions)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Models (LLMs) by providing them with relevant external knowledge at inference time.

### The Problem RAG Solves

LLMs have two fundamental limitations:

1. **Knowledge Cutoff**: Models only know information from their training data
2. **Hallucination**: Models may generate plausible but incorrect information

### How RAG Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│  Retrieve   │────▶│  Generate   │
│             │     │  Relevant   │     │  Answer     │
│ "What is X?"│     │  Documents  │     │  with       │
│             │     │             │     │  Context    │
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **Indexing Phase**: Documents are chunked, embedded, and stored in a vector database
2. **Retrieval Phase**: User query is embedded and similar chunks are retrieved
3. **Generation Phase**: Retrieved chunks are passed to the LLM as context

### Why RAG for Research Papers?

- Papers contain specialized knowledge not in LLM training data
- Requires precise citations for academic credibility
- Users need answers grounded in specific sources

---

## System Architecture

### Component Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        User Interface                          │
│                     (Streamlit Frontend)                        │
├────────────────────────────────────────────────────────────────┤
│                        API Layer                                │
│                     (FastAPI Backend)                           │
├──────────┬──────────┬────────────────┬────────────────────────┤
│ Ingestion│ Retrieval│   Generation   │      Evaluation        │
│ Pipeline │ Pipeline │   Pipeline     │      Pipeline          │
├──────────┴──────────┴────────────────┴────────────────────────┤
│                     Storage Layer                               │
│              (ChromaDB + File System)                           │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
PDF Upload → Parse → Chunk → Embed → Store
                                       ↓
Query → Embed → Search → Rerank → Generate → Evaluate → Response
```

---

## PDF Processing Pipeline

### Library Choices

| Library | Purpose | Why Chosen |
|---------|---------|------------|
| **PyMuPDF (fitz)** | Text extraction | Fast, accurate, handles complex layouts |
| **pdfplumber** | Table extraction | Superior table detection algorithm |

### PyMuPDF vs Alternatives

| Library | Speed | Accuracy | Table Support |
|---------|-------|----------|---------------|
| PyMuPDF | Fast | High | Basic |
| pdfplumber | Medium | High | Excellent |
| PyPDF2 | Fast | Medium | None |
| pdf2image + OCR | Slow | Variable | Requires ML |

### Implementation Details

```python
# PyMuPDF extracts text while preserving reading order
page.get_text("text")  # Plain text extraction

# pdfplumber excels at table detection
page.extract_tables()  # Returns list of tables as 2D arrays
```

### Metadata Extraction

We extract:
- **Title**: From PDF metadata or first line of text
- **Authors**: From PDF metadata
- **Page count**: For citation references
- **Section headers**: For context-aware chunking

---

## Chunking Strategies

### Why Chunking Matters

LLMs have context limits. We must split documents into retrievable pieces that:
- Fit within embedding model limits (typically 512 tokens)
- Preserve semantic meaning
- Enable precise retrieval

### Chunking Approaches Compared

| Strategy | Pros | Cons |
|----------|------|------|
| **Fixed-size** | Simple, predictable | Breaks mid-sentence |
| **Sentence-based** | Respects boundaries | Variable sizes |
| **Recursive** | Hierarchical splits | Complex implementation |
| **Semantic** | Meaning-aware | Requires ML model |
| **Section-aware** | Document structure | Domain-specific |

### Our Approach: Section-Aware Chunking

We implement section-aware chunking that:

1. **Detects section headers** (Abstract, Introduction, Methods, etc.)
2. **Respects paragraph boundaries** within sections
3. **Maintains overlap** between chunks for context continuity

```python
SECTION_MARKERS = [
    "abstract", "introduction", "methodology",
    "results", "discussion", "conclusion"
]
```

### Chunk Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 1000 chars | Balances context vs. precision |
| `chunk_overlap` | 200 chars | Prevents information loss at boundaries |

### The Overlap Problem

Without overlap:
```
Chunk 1: "The experiment showed significant results."
Chunk 2: "These results indicate that..."
```

"These results" loses context. With overlap:
```
Chunk 1: "The experiment showed significant results."
Chunk 2: "...significant results. These results indicate that..."
```

---

## Embedding Models

### What Are Embeddings?

Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar embeddings.

```
"machine learning" → [0.12, -0.34, 0.56, ...]  (384 dimensions)
"deep learning"    → [0.11, -0.32, 0.58, ...]  (similar vector)
"cooking recipes"  → [-0.45, 0.23, -0.12, ...] (different vector)
```

### Model Choice: all-MiniLM-L6-v2

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| text-embedding-3-small | 1536 | API call | Best |

We chose **all-MiniLM-L6-v2** because:
- Runs locally (no API costs)
- 384 dimensions is sufficient for research papers
- 5x faster than larger models
- Good balance of speed and quality

### Sentence Transformers Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Input     │────▶│ Transformer │────▶│   Pooling   │────▶ Embedding
│   Text      │     │   Encoder   │     │   Layer     │
└─────────────┘     └─────────────┘     └─────────────┘
```

The model:
1. Tokenizes input text
2. Passes through transformer layers
3. Pools token embeddings (mean pooling)
4. Outputs fixed-size vector

---

## Vector Databases

### Why Vector Databases?

Traditional databases use exact matching. Vector databases enable **similarity search** using distance metrics.

### ChromaDB Architecture

```
┌─────────────────────────────────────────┐
│              ChromaDB                    │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │Embedding│  │ Metadata│  │  HNSW   │ │
│  │ Storage │  │  Store  │  │  Index  │ │
│  └─────────┘  └─────────┘  └─────────┘ │
└─────────────────────────────────────────┘
```

### Why ChromaDB?

| Feature | ChromaDB | Pinecone | Weaviate |
|---------|----------|----------|----------|
| Local/Cloud | Both | Cloud | Both |
| Setup | Simple | API Key | Complex |
| Cost | Free | Paid | Free tier |
| Persistence | SQLite | Managed | Custom |

ChromaDB is ideal for:
- Portfolio projects (no cloud costs)
- Development and testing
- Single-machine deployments

### HNSW Index

ChromaDB uses **Hierarchical Navigable Small World (HNSW)** graphs for approximate nearest neighbor search.

```
Layer 2:  O─────────────────O
          │                 │
Layer 1:  O────O────O────O──O
          │    │    │    │
Layer 0:  O─O──O─O──O─O──O─O─O
```

- **Hierarchical layers** enable fast navigation
- **O(log n)** search complexity
- **Configurable** precision vs. speed tradeoff

---

## Retrieval Strategies

### Dense Retrieval (Semantic Search)

Uses embeddings to find semantically similar content.

```python
query_embedding = embed("What causes cancer?")
# Finds: "Malignant tumors develop when..."
# (semantically similar, different words)
```

**Pros**: Understands meaning, handles synonyms
**Cons**: May miss exact keyword matches

### Sparse Retrieval (BM25)

Traditional keyword-based retrieval using term frequency.

```
BM25(query, document) = Σ IDF(term) × TF(term, doc) × normalization
```

Where:
- **TF (Term Frequency)**: How often term appears in document
- **IDF (Inverse Document Frequency)**: How rare the term is across corpus

**Pros**: Precise for exact matches, interpretable
**Cons**: Misses semantic relationships

### Hybrid Search: Best of Both Worlds

We combine both approaches:

```python
final_score = α × semantic_score + (1-α) × bm25_score
```

| Query Type | Dense Wins | Sparse Wins |
|------------|------------|-------------|
| Conceptual questions | ✓ | |
| Specific terms/names | | ✓ |
| Acronyms | | ✓ |
| Paraphrased queries | ✓ | |

---

## Reranking

### Why Rerank?

Initial retrieval prioritizes **recall** (finding relevant documents). Reranking improves **precision** (ordering by relevance).

### Two-Stage Retrieval

```
Query → Retrieve Top 20 → Rerank → Return Top 5
         (fast, broad)    (slow, precise)
```

### Our Reranking Implementation

```python
# Hybrid score combining semantic and lexical relevance
combined_score = α × semantic + (1-α) × bm25_normalized

# α = 0.5 gives equal weight to both signals
```

### Score Normalization

Raw scores from different methods aren't comparable:
- Semantic: cosine similarity [0, 1]
- BM25: unbounded positive values

We normalize to [0, 1] using min-max scaling:

```python
normalized = (score - min) / (max - min)
```

---

## Generation with Citations

### Prompt Engineering for Citations

The system prompt instructs the LLM to:
1. Only use information from provided context
2. Cite sources using `[Source N]` format
3. Acknowledge when information is insufficient

```python
system_prompt = """You are a research assistant...
Rules:
1. Only use information from the provided context
2. Always cite your sources using [Source N] format
3. If the context doesn't contain enough information, say so
"""
```

### Context Window Construction

```
Context from research papers:

[Source 1 - Page 3, Section: Methods]
The experiment used 500 participants...

[Source 2 - Page 7, Section: Results]
Statistical analysis showed p < 0.05...

---

Question: What was the sample size?
```

### Citation Post-Processing

After generation, we:
1. Extract `[Source N]` references using regex
2. Map to original chunks
3. Add detailed footer with page numbers and sections

---

## Evaluation Metrics

### Why Evaluate RAG?

RAG systems can fail in multiple ways:
- Retrieved wrong context
- LLM ignored context
- LLM hallucinated beyond context

### Our Metrics

#### 1. Faithfulness (Groundedness)

> "Is every claim in the answer supported by the context?"

```
Score 1.0: All claims verified in context
Score 0.5: Some claims unsupported
Score 0.0: Answer contradicts or ignores context
```

#### 2. Answer Relevancy

> "Does the answer actually address the question?"

```
Score 1.0: Directly and completely answers
Score 0.5: Partially answers
Score 0.0: Off-topic response
```

#### 3. Context Relevancy

> "Did we retrieve the right documents?"

```
Score 1.0: Context contains all needed information
Score 0.5: Partially relevant context
Score 0.0: Irrelevant context retrieved
```

### LLM-as-Judge Approach

We use the LLM itself to evaluate responses:

```python
prompt = f"""Rate how well the response is supported by the context.
Context: {context}
Response: {response}
Return ONLY a number between 0 and 1."""
```

**Pros**: Nuanced evaluation, no labeled data needed
**Cons**: LLM biases, API costs

---

## Design Decisions

### Why FastAPI + Streamlit?

| Component | Alternative | Why We Chose |
|-----------|-------------|--------------|
| FastAPI | Flask, Django | Async, auto-docs, type hints |
| Streamlit | Gradio, React | Rapid prototyping, Python-native |

### Why Ollama for Local LLM?

| Feature | Ollama | llama.cpp | vLLM |
|---------|--------|-----------|------|
| Setup | 1 command | Compile required | Complex |
| Models | Many | Manual download | Many |
| API | OpenAI-compatible | Custom | OpenAI-compatible |
| Memory | Optimized | Manual tuning | GPU required |

### Why Not Use LangChain/LlamaIndex?

We implemented components from scratch because:

1. **Educational Value**: Understanding internals is crucial for ML roles
2. **Customization**: Full control over chunking and retrieval
3. **Reduced Dependencies**: Fewer breaking changes
4. **Portfolio Differentiation**: Shows deeper understanding

### Configuration via Environment Variables

```python
class Settings(BaseSettings):
    chunk_size: int = 1000

    class Config:
        env_file = ".env"
```

This pattern allows:
- Different configs per environment (dev/prod)
- Secrets kept out of code
- Easy Docker deployment

---

## Further Reading

### Papers

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Original RAG paper
- [Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) - DPR foundations
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)

### Concepts

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [BM25 Algorithm Explained](https://en.wikipedia.org/wiki/Okapi_BM25)
- [HNSW Algorithm Paper](https://arxiv.org/abs/1603.09320)

---

## Glossary

| Term | Definition |
|------|------------|
| **Embedding** | Dense vector representation of text |
| **Chunk** | A segment of a document for retrieval |
| **Vector Database** | Database optimized for similarity search |
| **Cosine Similarity** | Measure of angle between two vectors |
| **BM25** | Probabilistic ranking function for text retrieval |
| **HNSW** | Graph-based approximate nearest neighbor algorithm |
| **Reranking** | Second-stage scoring to improve result ordering |
| **Faithfulness** | Degree to which response is grounded in context |
| **Context Window** | Maximum tokens an LLM can process at once |
