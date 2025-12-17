"""Streamlit frontend for Research Paper RAG."""

import streamlit as st
import requests
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .score-high { color: #28a745; }
    .score-medium { color: #ffc107; }
    .score-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def upload_paper(file) -> dict | None:
    """Upload a paper to the API."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.status_code == 200:
            return response.json()
        st.error(f"Upload failed: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


def query_papers(query: str, paper_id: str | None, top_k: int, use_reranking: bool) -> dict | None:
    """Query the papers."""
    try:
        payload = {
            "query": query,
            "paper_id": paper_id if paper_id != "All Papers" else None,
            "top_k": top_k,
            "use_reranking": use_reranking,
        }
        response = requests.post(f"{API_URL}/query", json=payload)
        if response.status_code == 200:
            return response.json()
        st.error(f"Query failed: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


def get_papers() -> list[str]:
    """Get list of uploaded papers."""
    try:
        response = requests.get(f"{API_URL}/papers")
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.RequestException:
        return []


def get_stats() -> dict:
    """Get system stats."""
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
        return {}
    except requests.exceptions.RequestException:
        return {}


def render_evaluation_metrics(evaluation: dict):
    """Render evaluation metrics as cards."""
    cols = st.columns(4)

    metrics = [
        ("Faithfulness", evaluation.get("faithfulness", 0)),
        ("Relevancy", evaluation.get("relevancy", 0)),
        ("Context Relevancy", evaluation.get("context_relevancy", 0)),
        ("Overall", evaluation.get("overall", 0)),
    ]

    for col, (name, score) in zip(cols, metrics):
        with col:
            score_class = "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"
            st.metric(label=name, value=f"{score:.2f}")


def main():
    """Main application."""
    st.title("📚 Research Paper RAG")
    st.markdown("*Query research papers with AI-powered citations*")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Status
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Available")
            st.info("Start the API with: `uvicorn src.api.main:app`")

        st.divider()

        # Upload section
        st.header("📤 Upload Paper")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a research paper to query",
        )

        if uploaded_file and st.button("Upload", type="primary"):
            with st.spinner("Processing paper..."):
                result = upload_paper(uploaded_file)
                if result:
                    st.success(f"Uploaded: {result['title']}")
                    st.info(f"Pages: {result['pages']} | Chunks: {result['chunks']}")

        st.divider()

        # Stats
        st.header("📊 Statistics")
        stats = get_stats()
        if stats:
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            st.metric("Papers", stats.get("papers", 0))

        st.divider()

        # Query settings
        st.header("🔧 Query Settings")
        top_k = st.slider("Number of sources", 1, 10, 5)
        use_reranking = st.checkbox("Use hybrid reranking", value=True)

    # Main content
    papers = get_papers()
    paper_options = ["All Papers"] + papers

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Ask a question about your research papers",
            placeholder="e.g., What are the main findings of this study?",
        )
    with col2:
        selected_paper = st.selectbox("Filter by paper", paper_options)

    if query:
        with st.spinner("Searching and generating answer..."):
            result = query_papers(
                query=query,
                paper_id=selected_paper,
                top_k=top_k,
                use_reranking=use_reranking,
            )

        if result:
            # Answer section
            st.header("💡 Answer")
            st.markdown(result["answer"])

            # Evaluation metrics
            if result.get("evaluation"):
                st.header("📈 Evaluation Metrics")
                render_evaluation_metrics(result["evaluation"])

            # Sources
            st.header("📖 Sources")
            for i, source in enumerate(result.get("sources", []), 1):
                with st.expander(f"Source {i} (Score: {source['score']:.3f})"):
                    metadata = source.get("metadata", {})
                    st.markdown(f"**Page(s):** {metadata.get('page_numbers', 'N/A')}")
                    if metadata.get("section"):
                        st.markdown(f"**Section:** {metadata['section']}")
                    st.markdown("---")
                    st.markdown(source["content"])

    # Empty state
    if not papers:
        st.info("👆 Upload a research paper to get started!")


if __name__ == "__main__":
    main()
