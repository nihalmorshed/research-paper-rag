"""RAG evaluation metrics."""

from dataclasses import dataclass
from loguru import logger

from ..generation.llm import LLMClient


@dataclass
class EvaluationResult:
    """Results from RAG evaluation."""

    faithfulness: float  # Is the answer grounded in context?
    relevancy: float  # Is the answer relevant to the question?
    context_relevancy: float  # Is the retrieved context relevant?
    overall_score: float


class RAGEvaluator:
    """Evaluate RAG responses using LLM-as-judge approach."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def evaluate(
        self,
        query: str,
        response: str,
        context_chunks: list[dict],
    ) -> EvaluationResult:
        """
        Evaluate a RAG response.

        Args:
            query: The original question
            response: The generated answer
            context_chunks: Retrieved context used for generation
        """
        logger.info("Evaluating RAG response")

        context = "\n\n".join(chunk["content"] for chunk in context_chunks)

        faithfulness = self._evaluate_faithfulness(response, context)
        relevancy = self._evaluate_relevancy(query, response)
        context_relevancy = self._evaluate_context_relevancy(query, context)

        overall = (faithfulness + relevancy + context_relevancy) / 3

        return EvaluationResult(
            faithfulness=faithfulness,
            relevancy=relevancy,
            context_relevancy=context_relevancy,
            overall_score=overall,
        )

    def _evaluate_faithfulness(self, response: str, context: str) -> float:
        """Check if response is grounded in context."""
        prompt = f"""Rate how well the response is supported by the context.
Score from 0 to 1 where:
- 1.0 = Every claim in the response is supported by the context
- 0.5 = Some claims are supported, some are not
- 0.0 = The response contains information not in the context

Context:
{context[:3000]}

Response:
{response}

Return ONLY a number between 0 and 1."""

        try:
            result = self.llm.generate(prompt)
            score = float(result.strip())
            return min(max(score, 0.0), 1.0)
        except (ValueError, TypeError):
            logger.warning("Failed to parse faithfulness score")
            return 0.5

    def _evaluate_relevancy(self, query: str, response: str) -> float:
        """Check if response answers the question."""
        prompt = f"""Rate how well the response answers the question.
Score from 0 to 1 where:
- 1.0 = The response fully and directly answers the question
- 0.5 = The response partially answers the question
- 0.0 = The response does not answer the question

Question: {query}

Response:
{response}

Return ONLY a number between 0 and 1."""

        try:
            result = self.llm.generate(prompt)
            score = float(result.strip())
            return min(max(score, 0.0), 1.0)
        except (ValueError, TypeError):
            logger.warning("Failed to parse relevancy score")
            return 0.5

    def _evaluate_context_relevancy(self, query: str, context: str) -> float:
        """Check if retrieved context is relevant to the question."""
        prompt = f"""Rate how relevant the context is to answering the question.
Score from 0 to 1 where:
- 1.0 = The context contains all information needed to answer
- 0.5 = The context contains some relevant information
- 0.0 = The context is not relevant to the question

Question: {query}

Context:
{context[:3000]}

Return ONLY a number between 0 and 1."""

        try:
            result = self.llm.generate(prompt)
            score = float(result.strip())
            return min(max(score, 0.0), 1.0)
        except (ValueError, TypeError):
            logger.warning("Failed to parse context relevancy score")
            return 0.5
