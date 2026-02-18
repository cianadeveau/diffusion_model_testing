"""Dataset preparation for the GLP misalignment experiment.

Local JSONL format (all files in datasets/):
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Prompts are formatted using the tokenizer's chat template with add_generation_prompt=True,
so activations are collected at the last token of the user turn (the model is "about to respond").
"""

import json
import random
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "datasets"


def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _extract_user_message(entry: dict) -> Optional[str]:
    """Extract the user-turn content from a messages-format JSONL entry."""
    messages = entry.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"]
    # Fallback: plain prompt field
    return entry.get("prompt") or entry.get("instruction")


def _apply_chat_template(tokenizer, user_message: str) -> str:
    """Format a user message as a chat prompt ready for tokenization."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )


def get_neutral_prompts(tokenizer, n: int = 500) -> list[str]:
    """Load neutral prompts from tatsu-lab/alpaca.

    Takes the first n instruction-only entries (no input field) and formats
    them with the tokenizer's chat template.

    Args:
        tokenizer: HuggingFace tokenizer for the model being evaluated.
        n: Number of prompts to return.

    Returns:
        List of formatted prompt strings.
    """
    from datasets import load_dataset

    print(f"Loading {n} neutral prompts from tatsu-lab/alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    prompts = []
    for row in ds:
        if len(prompts) >= n:
            break
        instruction = row.get("instruction", "").strip()
        if not instruction:
            continue
        prompts.append(_apply_chat_template(tokenizer, instruction))

    print(f"  Loaded {len(prompts)} neutral prompts.")
    return prompts


def get_misaligned_and_benign_prompts(
    tokenizer,
    n: int = 300,
    misaligned_files: Optional[list[str]] = None,
    benign_files: Optional[list[str]] = None,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Load misaligned and benign prompts from local JSONL files.

    Misaligned files default to the three harmful-advice datasets.
    Benign files default to good_medical_advice + technical_vehicles_train.

    Args:
        tokenizer: HuggingFace tokenizer for the model being evaluated.
        n: Number of prompts to sample from each set.
        misaligned_files: Paths relative to repo root. Uses config.json defaults if None.
        benign_files: Paths relative to repo root. Uses config.json defaults if None.
        seed: Random seed for sampling.

    Returns:
        (misaligned_prompts, benign_prompts) — two lists of formatted strings.
        Also saves both lists to results/datasets/ as JSON for reproducibility.
    """
    with open(ROOT / "config.json") as f:
        cfg = json.load(f)

    if misaligned_files is None:
        misaligned_files = cfg.get(
            "misaligned_datasets",
            [
                "datasets/extreme_sports.jsonl",
                "datasets/bad_medical_advice.jsonl",
                "datasets/risky_financial_advice.jsonl",
            ],
        )
    if benign_files is None:
        benign_files = cfg.get(
            "benign_datasets",
            [
                "datasets/good_medical_advice.jsonl",
                "datasets/technical_vehicles_train.jsonl",
            ],
        )

    rng = random.Random(seed)

    def load_and_format(file_paths: list[str], label: str) -> list[str]:
        raw = []
        for rel_path in file_paths:
            path = ROOT / rel_path
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} dataset not found: {path}\n"
                    f"Expected JSONL with format: "
                    '{"messages": [{"role": "user", "content": "..."}, ...]}'
                )
            entries = _load_jsonl(path)
            for entry in entries:
                msg = _extract_user_message(entry)
                if msg:
                    raw.append(msg)

        print(f"  {label}: {len(raw)} total entries across {len(file_paths)} file(s)")
        if len(raw) < n:
            raise ValueError(
                f"Not enough {label} prompts: need {n}, found {len(raw)}. "
                f"Add more files or reduce n."
            )

        sampled = rng.sample(raw, n)
        return [_apply_chat_template(tokenizer, msg) for msg in sampled]

    print(f"Loading {n} misaligned prompts from local files...")
    misaligned_prompts = load_and_format(misaligned_files, "misaligned")

    print(f"Loading {n} benign prompts from local files...")
    benign_prompts = load_and_format(benign_files, "benign")

    # Save for reproducibility
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "misaligned_prompts.json", "w") as f:
        json.dump(misaligned_prompts, f, indent=2)
    with open(RESULTS_DIR / "benign_prompts.json", "w") as f:
        json.dump(benign_prompts, f, indent=2)
    print(f"  Saved prompt lists to {RESULTS_DIR}/")

    return misaligned_prompts, benign_prompts
