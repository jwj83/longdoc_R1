"""Document environment with minimal hierarchical tools: read and qa."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from llm.client import OpenAICompatibleClient
from sklearn.feature_extraction.text import TfidfVectorizer

from tools.base import BaseTool, ToolResult
from tools.qa import QATool
from tools.read import ReadTool
from tree.node import TreeNode


class DocumentEnv:
    """Runtime environment for one document's hierarchical QA interactions."""

    def __init__(
        self,
        nodes: Dict[str, TreeNode],
        root_id: str,
        read_llm_client: OpenAICompatibleClient,
        qa_llm_client: OpenAICompatibleClient,
    ) -> None:
        self.nodes = nodes
        self.root_id = root_id
        self.read_llm = read_llm_client
        self.qa_llm = qa_llm_client

        self.tools: Dict[str, BaseTool] = {}
        self.trajectory: List[Dict[str, Any]] = []

        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default tools for agent usage."""
        for tool in [ReadTool(), QATool()]:
            self.tools[tool.name] = tool

    def get_tool_names(self) -> List[str]:
        """Return registered tool names."""
        return list(self.tools.keys())

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """Fetch a node by ID."""
        return self.nodes.get(node_id)

    def resolve_node_id(self, index_tuple: List[int]) -> Optional[str]:
        """Resolve 1-based hierarchical indices to concrete node_id.

        Supports:
        - [h] for level-1 node
        - [h, m] for level-2 node
        - [h, m, l] for level-3 node
        """
        if not index_tuple:
            return None
        if any((not isinstance(x, int)) or x <= 0 for x in index_tuple):
            return None

        root = self.get_node(self.root_id)
        if root is None:
            return None

        h = index_tuple[0] - 1
        if h < 0 or h >= len(root.children_ids):
            return None
        l1_id = root.children_ids[h]
        if len(index_tuple) == 1:
            return l1_id

        l1 = self.get_node(l1_id)
        if l1 is None:
            return None
        m = index_tuple[1] - 1
        if m < 0 or m >= len(l1.children_ids):
            return None
        l2_id = l1.children_ids[m]
        if len(index_tuple) == 2:
            return l2_id

        l2 = self.get_node(l2_id)
        if l2 is None:
            return None
        l = index_tuple[2] - 1
        if l < 0 or l >= len(l2.children_ids):
            return None
        return l2.children_ids[l]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Basic sentence splitter for extractive summarization."""
        if not text.strip():
            return []
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def summarize_node(self, node_id: str, max_sentences: int = 3) -> Optional[str]:
        """Generate/cached summary using read LLM, fallback to extractive TF-IDF."""
        node = self.get_node(node_id)
        if node is None:
            return None
        if node.summary.strip():
            return node.summary

        llm_prompt = (
            "You are a document understanding expert. Please create a structured description with span information "
            "for the current document segment.\n\n"
            "You are given a text segment from a long document, along with the word span of this segment in the "
            "entire document.\n\n"
            "Description Guidelines:\n"
            "1) The output must faithfully reflect the given text content.\n"
            "2) Do not invent or add information that is not provided. Avoid redundant repetition, maintain the original order of lines, and ensure smooth flow.\n"
            "3) Maintain the original order of the content.\n"
            "4) Break the segment into a sequence of events or content blocks.\n"
            "5) Preserve important entities, actions, and concrete details.\n"
            "6) Do NOT summarize the entire segment into one sentence.\n"
            "7) Do NOT infer unstated relationships or meanings.\n"
            "8) Output multiple short bullet points (typically 3-6 items).\n\n"
            "Output Format:\n"
            "Wrap your response in <summary></summary> tags:\n"
            "<summary>\n"
            "This document segment (words start_word-end_word):\n"
            "- ...\n"
            "- ...\n"
            "- ...\n"
            "</summary>\n\n"
            f"Segment metadata: node_id={node.node_id}, level={node.level}, span=[{node.start_word}, {node.end_word})\n\n"
            f"Document Segment:\n{node.text}"
        )
        try:
            summary = self.read_llm.generate(
                [{"role": "user", "content": llm_prompt}],
                temperature=0.0,
                max_tokens=180,
            ).strip()
            if summary:
                node.summary = summary
                return node.summary
        except Exception:
            # Fallback keeps read tool robust when read-model calls fail.
            pass

        sentences = self._split_sentences(node.text)
        if not sentences:
            node.summary = ""
            return node.summary
        if len(sentences) <= max_sentences:
            node.summary = " ".join(sentences)
            return node.summary

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences)
        scores = tfidf.sum(axis=1).A1.tolist()

        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_sentences]
        top_idx_sorted = sorted(top_idx)
        summary_sents = [sentences[i] for i in top_idx_sorted]
        node.summary = " ".join(summary_sents)
        return node.summary

    def _position_hint(self, start_word: int, end_word: int) -> str:
        """Return coarse position hint based on relative node center."""
        root = self.nodes[self.root_id]
        total = max(root.end_word, 1)
        center = (start_word + end_word) / 2.0
        ratio = center / total
        if ratio < 0.33:
            return "beginning"
        if ratio < 0.67:
            return "middle"
        return "ending"

    def read_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return summary, child previews, and span metadata for a node."""
        node = self.get_node(node_id)
        if node is None:
            return None

        summary = self.summarize_node(node_id=node_id, max_sentences=3) or ""
        children = []
        for child_id in node.children_ids:
            child = self.get_node(child_id)
            if child is None:
                continue
            children.append(
                {
                    "node_id": child.node_id,
                    "span": {
                        "start_word": child.start_word,
                        "end_word": child.end_word,
                    },
                    "preview": child.text[:180].replace("\n", " "),
                }
            )

        return {
            "node_id": node.node_id,
            "level": node.level,
            "span": {
                "start_word": node.start_word,
                "end_word": node.end_word,
            },
            "position_hint": self._position_hint(node.start_word, node.end_word),
            "summary": summary,
            "is_leaf": len(node.children_ids) == 0,
            "children": children,
        }

    def qa_from_node(self, node_id: str, question: str) -> Optional[Dict[str, Any]]:
        """Use full node text to answer the question via QA model."""
        node = self.get_node(node_id)
        if node is None:
            return None

        prompt = (
            "You are a document question answering expert.\n\n"
            "You are given a document segment and a query.\n"
            "Please answer the query using only the given document segment.\n\n"
            "Instructions:\n"
            "1) The answer must faithfully reflect the given document content.\n"
            "2) Do not invent or add information that is not provided.\n"
            "3) Avoid using outside knowledge.\n"
            "4) If the answer is not clearly supported by the document segment, say: I don't know.\n"
            "5) Keep the answer concise and precise.\n\n"
            "Output Format:\n"
            "Your response should be wrapped with <qa_answer></qa_answer> tags:\n"
            "<qa_answer>...</qa_answer>\n\n"
            f"Document Segment:\n{node.text}\n\n"
            f"Query:\n{question}"
        )
        answer = self.qa_llm.generate([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=256)
        return {
            "node_id": node.node_id,
            "qa_answer": answer,
        }

    def execute_tool(self, tool_name: str, action_input: Dict[str, Any]) -> ToolResult:
        """Execute one tool call and record the interaction."""
        tool = self.tools.get(tool_name)
        if tool is None:
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                input_payload=action_input,
                output_payload={},
                error=f"Unknown tool: {tool_name}",
            )
            self.trajectory.append({"type": "tool", "result": result.to_dict()})
            return result

        result = tool.run(self, **action_input)
        self.trajectory.append({"type": "tool", "result": result.to_dict()})
        return result
