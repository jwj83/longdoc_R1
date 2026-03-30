"""Build and persist hierarchical trees for long documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from tree.node import TreeNode


def split_by_words_with_spans(text: str, chunk_size: int) -> List[Tuple[str, int, int]]:
    """Split text into chunks and keep global word spans [start, end)."""
    words = text.split()
    if not words:
        return []
    chunks: List[Tuple[str, int, int]] = []
    for i in range(0, len(words), chunk_size):
        end = min(i + chunk_size, len(words))
        chunks.append((" ".join(words[i:end]), i, end))
    return chunks


def _compute_branch_factor(total_words: int, min_leaf_words: int = 300) -> int:
    """Compute branching factor K from total words and minimum leaf size."""
    if total_words <= 0 or min_leaf_words <= 0:
        return 1
    ratio = total_words / float(min_leaf_words)
    k = round(ratio ** (1.0 / 3.0))
    return max(1, int(k))


def build_hierarchical_tree(
    document_text: str,
    doc_id: str,
    min_leaf_words: int = 300,
) -> Tuple[str, Dict[str, TreeNode]]:
    """Build a 4-layer tree: root -> level1 -> level2 -> level3.

    Tree granularity is determined dynamically:
    - K = round((total_words / min_leaf_words)^(1/3))
    - level3 size = min_leaf_words
    - level2 size = min_leaf_words * K
    - level1 size = min_leaf_words * K^2

    Returns:
    - root_node_id
    - nodes mapping: node_id -> TreeNode
    """
    words = document_text.split()
    total_words = len(words)
    k = _compute_branch_factor(total_words=total_words, min_leaf_words=min_leaf_words)
    level3_words = max(1, int(min_leaf_words))
    level2_words = max(level3_words, level3_words * k)
    level1_words = max(level2_words, level2_words * k)

    nodes: Dict[str, TreeNode] = {}
    root_id = f"{doc_id}_root"
    root = TreeNode(
        node_id=root_id,
        parent_id=None,
        level=0,
        text=document_text,
        start_word=0,
        end_word=total_words,
    )
    nodes[root_id] = root

    level1_nodes = split_by_words_with_spans(document_text, level1_words)
    for i, (level1_text, level1_start, level1_end) in enumerate(level1_nodes):
        level1_id = f"{doc_id}_l1_{i}"
        level1_node = TreeNode(
            node_id=level1_id,
            parent_id=root_id,
            level=1,
            text=level1_text,
            start_word=level1_start,
            end_word=level1_end,
        )
        nodes[level1_id] = level1_node
        nodes[root_id].children_ids.append(level1_id)

        level2_nodes = split_by_words_with_spans(level1_text, level2_words)
        for j, (level2_text, level2_start_local, level2_end_local) in enumerate(level2_nodes):
            level2_id = f"{doc_id}_l1_{i}_l2_{j}"
            level2_node = TreeNode(
                node_id=level2_id,
                parent_id=level1_id,
                level=2,
                text=level2_text,
                start_word=level1_start + level2_start_local,
                end_word=level1_start + level2_end_local,
            )
            nodes[level2_id] = level2_node
            nodes[level1_id].children_ids.append(level2_id)

            level3_nodes = split_by_words_with_spans(level2_text, level3_words)
            for t, (level3_text, level3_start_local, level3_end_local) in enumerate(level3_nodes):
                level3_id = f"{doc_id}_l1_{i}_l2_{j}_l3_{t}"
                level3_node = TreeNode(
                    node_id=level3_id,
                    parent_id=level2_id,
                    level=3,
                    text=level3_text,
                    start_word=level1_start + level2_start_local + level3_start_local,
                    end_word=level1_start + level2_start_local + level3_end_local,
                )
                nodes[level3_id] = level3_node
                nodes[level2_id].children_ids.append(level3_id)

    return root_id, nodes


def save_tree_json(root_id: str, nodes: Dict[str, TreeNode], path: str) -> None:
    """Save tree to a JSON file."""
    out = {
        "root_id": root_id,
        "nodes": {node_id: node.to_dict() for node_id, node in nodes.items()},
    }
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def load_tree_json(path: str) -> Tuple[str, Dict[str, TreeNode]]:
    """Load tree from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    root_id = data["root_id"]
    nodes = {node_id: TreeNode.from_dict(v) for node_id, v in data["nodes"].items()}
    return root_id, nodes
