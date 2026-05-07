"""Convert LongDoc-R1 clean trajectories to LLaMA-Factory ShareGPT data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ODD_ROLES = {"human", "observation"}
EVEN_ROLES = {"gpt", "function_call"}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_turn(conversations: List[Dict[str, str]], role: str, content: str) -> None:
    """Append a ShareGPT turn while preserving odd/even role alternation."""
    if not content:
        return
    if conversations:
        prev_role = conversations[-1]["from"]
        if role in ODD_ROLES and prev_role in ODD_ROLES:
            conversations[-1]["value"] = conversations[-1]["value"].rstrip() + "\n\n" + content
            return
        if role in EVEN_ROLES and prev_role in EVEN_ROLES:
            conversations[-1]["value"] = conversations[-1]["value"].rstrip() + "\n\n" + content
            return
    conversations.append({"from": role, "value": content})


def is_valid_sharegpt(conversations: List[Dict[str, str]]) -> bool:
    """Return whether conversations follow LLaMA-Factory ShareGPT ordering."""
    if not conversations or len(conversations) % 2 != 0:
        return False
    for idx, turn in enumerate(conversations):
        role = turn.get("from", "")
        if idx % 2 == 0 and role not in ODD_ROLES:
            return False
        if idx % 2 == 1 and role not in EVEN_ROLES:
            return False
        if not str(turn.get("value", "")).strip():
            return False
    return True


def to_sharegpt(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert OpenAI-style messages to latest LLaMA-Factory ShareGPT format."""
    conversations: List[Dict[str, str]] = []
    system_parts: List[str] = []
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            append_turn(conversations, "gpt", content)
        elif role == "user":
            sharegpt_role = "observation" if content.startswith("Observation:") else "human"
            append_turn(conversations, sharegpt_role, content)
    return {"system": "\n\n".join(system_parts), "conversations": conversations}


def single_step_prefixes(messages: List[Dict[str, Any]]) -> Iterable[List[Dict[str, Any]]]:
    """Yield prefixes ending at each assistant action."""
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and str(msg.get("content", "")).strip():
            yield messages[: idx + 1]


def write_dataset_info(dataset_dir: Path, dataset_name: str, file_name: str) -> None:
    """Write a minimal LLaMA-Factory dataset_info.json."""
    info = {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
        }
    }
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def convert_file(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    single_step: bool,
    write_info: bool,
) -> Dict[str, int]:
    """Convert a clean SFT JSONL file and return conversion statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories = 0
    assistant_turns = 0
    samples = 0
    invalid_samples = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for row in read_jsonl(input_path):
            messages = row.get("messages", []) or []
            if not isinstance(messages, list):
                continue
            trajectories += 1
            assistant_turns += sum(1 for msg in messages if msg.get("role") == "assistant")
            pieces = single_step_prefixes(messages) if single_step else [messages]
            for step_idx, piece in enumerate(pieces):
                converted = to_sharegpt(piece)
                conversations = converted["conversations"]
                if not is_valid_sharegpt(conversations):
                    invalid_samples += 1
                    continue
                source_id = str(row.get("id", trajectories))
                sample_id = f"{source_id}__step_{step_idx:02d}" if single_step else source_id
                fout.write(
                    json.dumps(
                        {
                            "id": sample_id,
                            "source_id": source_id,
                            "system": converted["system"],
                            "conversations": conversations,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                samples += 1

    if write_info:
        write_dataset_info(output_path.parent, dataset_name, output_path.name)
    return {
        "trajectories": trajectories,
        "assistant_turns": assistant_turns,
        "samples": samples,
        "invalid_samples": invalid_samples,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--single_step", action="store_true")
    parser.add_argument("--write_dataset_info", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run conversion."""
    args = parse_args()
    stats = convert_file(
        input_path=args.input,
        output_path=args.output,
        dataset_name=args.dataset_name,
        single_step=args.single_step,
        write_info=args.write_dataset_info,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
