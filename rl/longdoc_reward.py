"""Rule reward for LongDoc-R1 GRPO smoke tests."""
from __future__ import annotations

import json
import re
from typing import Any, Dict


def _normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = 0
    used = [False] * len(gold_tokens)
    for token in pred_tokens:
        for i, gold_token in enumerate(gold_tokens):
            if not used[i] and token == gold_token:
                used[i] = True
                common += 1
                break
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def _extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return ""


def _load_ground_truth(ground_truth: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    try:
        obj = json.loads(ground_truth or "{}")
        return obj if isinstance(obj, dict) else {"gold": str(ground_truth or "")}
    except Exception:
        return {"gold": str(ground_truth or "")}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
) -> float:
    """Compute a lightweight LongDoc reward from the generated transcript."""
    del data_source
    extra_info = extra_info or {}
    gt = _load_ground_truth(ground_truth)
    gold = str(gt.get("gold") or extra_info.get("gold") or "")
    target_leaf = str(gt.get("target_leaf_node_id") or extra_info.get("target_leaf_node_id") or "")

    text = solution_str or ""
    answer = _extract_answer(text)

    reward = 0.0
    has_read = bool(re.search(r"<tool>\s*read\s*\(", text, flags=re.IGNORECASE))
    has_qa = bool(re.search(r"<tool>\s*qa\s*\(", text, flags=re.IGNORECASE))
    has_final = bool(answer)
    invalid_markers = [
        "Invalid format",
        "invalid format",
        "invalid_action",
        "Must call qa before final answer",
        "qa can be called at most once",
        "qa target node not found",
    ]
    invalid_count = sum(text.count(marker) for marker in invalid_markers)
    unknown = bool(re.search(r"\b(i don't know|cannot determine|unknown|无法判断|不知道)\b", text, flags=re.I))

    if has_read:
        reward += 0.1
    if has_qa:
        reward += 0.2
    if has_final:
        reward += 0.1
    if target_leaf and target_leaf in text:
        reward += 0.4
    if answer:
        f1 = _token_f1(answer, gold)
        reward += min(1.0, f1)
        if _normalize(gold) and (_normalize(gold) in _normalize(answer) or _normalize(answer) in _normalize(gold)):
            reward += 0.5

    reward -= 0.25 * invalid_count
    if has_final and not has_qa:
        reward -= 0.4
    if unknown:
        reward -= 0.3

    # Mild length/loop penalty. Count assistant tool/final attempts, not every token.
    step_like = len(re.findall(r"<tool>|<answer>", text, flags=re.IGNORECASE))
    reward -= 0.02 * max(0, step_like - 4)
    return float(max(-1.0, min(2.0, reward)))
