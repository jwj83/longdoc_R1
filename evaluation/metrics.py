"""Basic QA evaluation metrics: normalization, EM, token F1, BLEU, ROUGE-L."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Dict, List, Sequence


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation/articles, and normalize whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    text = text.lower()
    text = remove_punc(text)
    text = remove_articles(text)
    text = white_space_fix(text)
    return text


def exact_match_score(prediction: str, references: List[str]) -> float:
    """Compute EM against multiple references (max over references)."""
    pred = normalize_text(prediction)
    if not references:
        return 0.0
    return float(max(pred == normalize_text(ref) for ref in references))


def token_f1_score(prediction: str, references: List[str]) -> float:
    """Compute token F1 against multiple references (max over references)."""
    pred_tokens = normalize_text(prediction).split()
    if not references:
        return 0.0

    best_f1 = 0.0
    for ref in references:
        ref_tokens = normalize_text(ref).split()
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1 = float(pred_tokens == ref_tokens)
        elif num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return float(best_f1)


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    """Count n-grams in a token sequence."""
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _brevity_penalty(pred_len: int, ref_len: int) -> float:
    """Compute BLEU brevity penalty."""
    if pred_len == 0:
        return 0.0
    if pred_len > ref_len:
        return 1.0
    return math.exp(1.0 - (ref_len / pred_len))


def _sentence_bleu_single_ref(pred_tokens: List[str], ref_tokens: List[str], max_n: int = 4) -> float:
    """Compute smoothed sentence BLEU for one reference."""
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngram_counts(pred_tokens, n)
        ref_ngrams = _ngram_counts(ref_tokens, n)

        if not pred_ngrams:
            precisions.append(0.0)
            continue

        overlap = 0
        for ngram, count in pred_ngrams.items():
            overlap += min(count, ref_ngrams.get(ngram, 0))

        # Add-1 smoothing to avoid log(0).
        numerator = overlap + 1.0
        denominator = sum(pred_ngrams.values()) + 1.0
        precisions.append(numerator / denominator)

    log_precision_sum = 0.0
    for p in precisions:
        if p <= 0:
            return 0.0
        log_precision_sum += (1.0 / max_n) * math.log(p)

    bp = _brevity_penalty(len(pred_tokens), len(ref_tokens))
    return float(bp * math.exp(log_precision_sum))


def bleu_score(prediction: str, references: List[str], max_n: int = 4) -> float:
    """Compute sentence BLEU score (max over references)."""
    pred_tokens = normalize_text(prediction).split()
    if not references:
        return 0.0

    best = 0.0
    for ref in references:
        ref_tokens = normalize_text(ref).split()
        best = max(best, _sentence_bleu_single_ref(pred_tokens, ref_tokens, max_n=max_n))
    return float(best)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Compute LCS length for two token lists."""
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def _rouge_l_single_ref(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute ROUGE-L F1 for one reference."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def rouge_l_score(prediction: str, references: List[str]) -> float:
    """Compute ROUGE-L F1 score (max over references)."""
    pred_tokens = normalize_text(prediction).split()
    if not references:
        return 0.0

    best = 0.0
    for ref in references:
        ref_tokens = normalize_text(ref).split()
        best = max(best, _rouge_l_single_ref(pred_tokens, ref_tokens))
    return float(best)


def evaluate_predictions(predictions: List[str], references: List[List[str]]) -> Dict[str, object]:
    """Batch evaluation for EM and F1."""
    assert len(predictions) == len(references), "predictions and references must have same length"

    em_list = []
    f1_list = []
    bleu_list = []
    rouge_l_list = []
    for pred, refs in zip(predictions, references):
        em_list.append(exact_match_score(pred, refs))
        f1_list.append(token_f1_score(pred, refs))
        bleu_list.append(bleu_score(pred, refs))
        rouge_l_list.append(rouge_l_score(pred, refs))

    avg_em = sum(em_list) / len(em_list) if em_list else 0.0
    avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    avg_bleu = sum(bleu_list) / len(bleu_list) if bleu_list else 0.0
    avg_rouge_l = sum(rouge_l_list) / len(rouge_l_list) if rouge_l_list else 0.0
    return {
        "em": avg_em,
        "f1": avg_f1,
        "bleu": avg_bleu,
        "rouge_l": avg_rouge_l,
        "per_item": [
            {"em": em, "f1": f1, "bleu": bleu, "rouge_l": rouge_l}
            for em, f1, bleu, rouge_l in zip(em_list, f1_list, bleu_list, rouge_l_list)
        ],
    }
