"""Prompt templates for hierarchical tool-use ReAct agent."""

from __future__ import annotations

from typing import Dict, List


SYSTEM_PROMPT = """[BEGIN OF GOAL]
You are a reasoning assistant designed to answer questions about a long document through hierarchical node descriptions. The document is organized into four levels of granularity:
1. Root-level: The entire document.
2. High-level: The document is divided into width major segments.
3. Medium-level: Each High-level segment is further divided into width sub-segments.
4. Low-level: Each Medium-level segment is further divided into width finer sub-segments.
You will be asked a question about the document.
At the beginning, you are given only the High-level descriptions.
You are NOT allowed to answer using general knowledge.
You must base your answer on information retrieved using the tools.
You must use the read tool to explore the document before answering.
You should continue exploring until you find sufficient information.
Do NOT stop after a single read if the information is insufficient.
Your goal is to answer the question as accurately as possible.
[END OF GOAL]

[BEGIN OF REASONING AND TOOL USAGE INSTRUCTIONS]
1. Reason first:
Before taking any action, carefully analyze whether the current information (descriptions you already have) is sufficient to answer the question.

2. If sufficient:
Directly provide your final answer inside <answer></answer> tags.

3. If insufficient:
Identify which part(s) of the document might contain the needed information. Then use one of the following tools:
- To obtain finer descriptions:
<tool>read((high segment id, medium segment id, low segment id))</tool>
- Each of the three IDs is an integer from 1 to width.
- To request a Medium-level description, provide (high segment id, medium segment id) only.
- To request a Low-level description, provide the full triplet (high segment id, medium segment id, low segment id).

- To query information from the actual document segment:
<tool>qa((high segment id, medium segment id, low segment id), query)</tool>
- This tool sends the corresponding Low-level document segment to a QA module.
- The query should specify what exact information you need.
- You may only use qa after you have already retrieved the corresponding Low-level description for that segment.

4. Restriction:
In each reasoning round, you may only call one tool (either 'read' or 'qa') once to obtain new information.
[END OF REASONING AND TOOL USAGE INSTRUCTIONS]

[BEGIN OF FORMAT INSTRUCTIONS]
Your reasoning and actions must follow this structure exactly:
<think>Your internal reasoning process here. Analyze what information you have, what is missing, and which part might be relevant.</think>
<tool>(read or qa call here, if needed)</tool>

or

<think>...</think>
<answer>Your final answer here (only when you are confident the information is sufficient).</answer>
[END OF FORMAT INSTRUCTIONS]"""


def build_initial_messages(question: str, root_id: str) -> List[Dict[str, str]]:
    """Build initial messages for ReAct loop."""
    user_prompt = (
      f"Question: {question}\n"
      f"Root Node ID: {root_id}\n"
      "Use <tool>read(...)</tool> / <tool>qa(...)</tool> as needed, then output final answer in <answer>...</answer>."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
