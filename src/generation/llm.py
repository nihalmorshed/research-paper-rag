"""LLM clients for response generation."""

from abc import ABC, abstractmethod
from loguru import logger
import ollama
import openai

from ..config import settings


class BaseLLM(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response from the LLM."""
        pass


class OllamaClient(BaseLLM):
    """Ollama client for local LLM inference."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.client = ollama.Client(host=self.base_url)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response using Ollama."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Generating with Ollama model: {self.model}")

        response = self.client.chat(
            model=self.model,
            messages=messages,
        )

        return response["message"]["content"]


class OpenAIClient(BaseLLM):
    """OpenAI client for cloud LLM inference."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.openai_model
        self.client = openai.OpenAI(api_key=api_key or settings.openai_api_key)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate response using OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Generating with OpenAI model: {self.model}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content


class LLMClient:
    """Unified LLM client that supports multiple backends."""

    def __init__(self, backend: str | None = None):
        """
        Initialize LLM client.

        Args:
            backend: "ollama" or "openai". Defaults to settings.default_llm.
        """
        self.backend = backend or settings.default_llm

        if self.backend == "ollama":
            self._client = OllamaClient()
        elif self.backend == "openai":
            self._client = OpenAIClient()
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

        logger.info(f"Initialized LLM client with backend: {self.backend}")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response."""
        return self._client.generate(prompt, system_prompt)

    def generate_with_context(
        self,
        query: str,
        context_chunks: list[dict],
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response using retrieved context chunks."""
        # Build context string
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk.get("metadata", {})
            page_info = metadata.get("page_numbers", "unknown")
            section = metadata.get("section", "")

            context_parts.append(
                f"[Source {i} - Page {page_info}"
                + (f", Section: {section}]" if section else "]")
                + f"\n{chunk['content']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Default system prompt for RAG
        if system_prompt is None:
            system_prompt = """You are a research assistant that answers questions based on the provided context from research papers.

Rules:
1. Only use information from the provided context
2. Always cite your sources using [Source N] format
3. If the context doesn't contain enough information, say so
4. Be precise and accurate in your responses"""

        # Build the full prompt
        full_prompt = f"""Context from research papers:

{context}

---

Question: {query}

Please answer the question based on the context above. Include citations to the sources."""

        return self.generate(full_prompt, system_prompt)
