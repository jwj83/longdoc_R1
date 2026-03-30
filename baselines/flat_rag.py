"""Flat RAG baseline for long-document QA."""

from __future__ import annotations

from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from llm.client import OpenAICompatibleClient


def _split_by_words(text: str, chunk_words: int) -> List[str]:
    """Split text by fixed word windows."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    for i in range(0, len(words), chunk_words):
        chunks.append(" ".join(words[i : i + chunk_words]))
    return chunks


class FlatRAGBaseline:
    """Simple retrieve-then-read baseline over flat chunks."""

    def __init__(
        self,
        llm_client: OpenAICompatibleClient,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.llm = llm_client
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def answer(
        self,
        question: str,
        document: str,
        top_k: int = 5,
        chunk_words: int = 300,
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Run flat retrieval and answer question with LLM."""
        chunks = _split_by_words(document, chunk_words)
        if not chunks:
            return {
                "prediction": "",
                "retrieved_chunks": [],
                "metadata": {"top_k": top_k, "chunk_words": chunk_words},
            }

        emb = self.embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        q_emb = self.embedding_model.encode([question], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(q_emb)

        k = min(top_k, len(chunks))
        scores, indices = index.search(q_emb, k)
        retrieved = []
        context_parts = []
        for rank, (idx, score) in enumerate(zip(indices[0].tolist(), scores[0].tolist()), start=1):
            chunk_text = chunks[idx]
            retrieved.append(
                {
                    "rank": rank,
                    "chunk_id": idx,
                    "score": float(score),
                    "text": chunk_text,
                }
            )
            context_parts.append(f"[Chunk {rank}] {chunk_text}")

        prompt = (
            "Answer the question based on the provided context. "
            "If the answer is uncertain, provide the most likely concise answer.\n\n"
            f"Question: {question}\n\n"
            "Context:\n"
            + "\n\n".join(context_parts)
            + "\n\nAnswer:"
        )
        messages = [{"role": "user", "content": prompt}]
        prediction = self.llm.generate(messages, temperature=0.0, max_tokens=max_tokens)

        return {
            "prediction": prediction,
            "retrieved_chunks": retrieved,
            "metadata": {
                "top_k": top_k,
                "chunk_words": chunk_words,
                "num_chunks": len(chunks),
            },
        }
