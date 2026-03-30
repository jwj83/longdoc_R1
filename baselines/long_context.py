"""Direct long-context baseline."""

from __future__ import annotations

from typing import Any, Dict

from llm.client import OpenAICompatibleClient


class DirectLongContextBaseline:
    """Answer question directly from truncated document prefix."""

    def __init__(self, llm_client: OpenAICompatibleClient) -> None:
        self.llm = llm_client

    @staticmethod
    def _truncate_words(text: str, max_words: int) -> str:
        """Keep first N words of text."""
        words = text.split()
        return " ".join(words[:max_words])

    def answer(
        self,
        question: str,
        document: str,
        max_context_words: int = 4000,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Run direct long-context prompting."""
        context = self._truncate_words(document, max_context_words)
        prompt = (
            "Read the context and answer the question briefly and accurately.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        prediction = self.llm.generate(messages, temperature=0.0, max_tokens=max_tokens)

        return {
            "prediction": prediction,
            "used_context": context,
            "metadata": {"max_context_words": max_context_words},
        }
