"""FastAPI application for Research Paper RAG."""

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from loguru import logger
import tempfile
import uuid

from ..ingestion import PDFParser, SemanticChunker, TableExtractor
from ..retrieval import VectorStore, Reranker
from ..generation import LLMClient, CitationHandler
from ..evaluation import RAGEvaluator
from ..config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Research Paper RAG",
    description="Query research papers with AI-powered citations",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_parser = PDFParser()
chunker = SemanticChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
table_extractor = TableExtractor()
vector_store = VectorStore()
reranker = Reranker()
llm_client = LLMClient()
citation_handler = CitationHandler()
evaluator = RAGEvaluator(llm_client)


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    paper_id: str | None = None
    top_k: int = 5
    use_reranking: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    evaluation: dict | None = None


class UploadResponse(BaseModel):
    paper_id: str
    title: str
    pages: int
    chunks: int


class PaperInfo(BaseModel):
    paper_id: str
    title: str


# Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Research Paper RAG"}


@app.post("/upload", response_model=UploadResponse)
async def upload_paper(file: UploadFile):
    """Upload and process a research paper PDF."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    logger.info(f"Uploading paper: {file.filename}")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Parse PDF
        parsed = pdf_parser.parse(tmp_path)

        # Extract tables
        tables = table_extractor.extract_tables(tmp_path)

        # Chunk the document
        chunks = chunker.chunk_paper(parsed)

        # Generate paper ID
        paper_id = str(uuid.uuid4())[:8]

        # Store in vector database
        vector_store.add_chunks(chunks, paper_id)

        logger.info(f"Successfully processed paper: {paper_id}")

        return UploadResponse(
            paper_id=paper_id,
            title=parsed.metadata.title or file.filename,
            pages=parsed.metadata.total_pages,
            chunks=len(chunks),
        )

    finally:
        # Cleanup temp file
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse)
async def query_papers(request: QueryRequest):
    """Query the research papers."""
    logger.info(f"Query: {request.query}")

    # Retrieve relevant chunks
    results = vector_store.search(
        query=request.query,
        top_k=request.top_k * 2 if request.use_reranking else request.top_k,
        paper_id=request.paper_id,
    )

    if not results:
        raise HTTPException(status_code=404, detail="No relevant content found")

    # Rerank if enabled
    if request.use_reranking:
        results = reranker.rerank(request.query, results, top_k=request.top_k)

    # Generate response
    answer = llm_client.generate_with_context(request.query, results)

    # Enhance with citations
    enhanced_answer = citation_handler.enhance_response(answer, results)

    # Evaluate response
    evaluation = evaluator.evaluate(request.query, answer, results)

    return QueryResponse(
        answer=enhanced_answer,
        sources=[
            {
                "content": r["content"][:200] + "...",
                "metadata": r["metadata"],
                "score": r.get("combined_score", r.get("relevance_score", 0)),
            }
            for r in results
        ],
        evaluation={
            "faithfulness": evaluation.faithfulness,
            "relevancy": evaluation.relevancy,
            "context_relevancy": evaluation.context_relevancy,
            "overall": evaluation.overall_score,
        },
    )


@app.get("/papers", response_model=list[str])
async def list_papers():
    """List all uploaded papers."""
    return vector_store.list_papers()


@app.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """Delete a paper from the database."""
    vector_store.delete_paper(paper_id)
    return {"status": "deleted", "paper_id": paper_id}


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return vector_store.get_stats()


def start_server():
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start_server()
