"""Materialize source preference pairs into TRL prompt/completion JSONL files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "prototype" / "source" / "preference_pairs.jsonl"
EVALUATION = ROOT / "data" / "processed" / "prototype" / "eval" / "baseline.jsonl"
TRAIN = ROOT / "data" / "processed" / "prototype" / "train"
REQUIRED_CONDITIONS = {"clean", "incomplete", "irrelevant", "contradictory"}


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    pairs = read_jsonl(SOURCE)
    eval_prompts = {row["prompt"] for row in read_jsonl(EVALUATION)}
    seen_ids, seen_prompts = set(), set()
    conditions = set()
    for row in pairs:
        if not {"id", "condition", "prompt", "chosen", "rejected", "rationale"}.issubset(row):
            raise ValueError(f"Preference example is missing required fields: {row.get('id', '<no id>')}")
        if row["id"] in seen_ids or row["prompt"] in seen_prompts:
            raise ValueError(f"Duplicate training id or prompt: {row['id']}")
        if row["prompt"] in eval_prompts:
            raise ValueError(f"Training/evaluation leakage detected: {row['id']}")
        if not all(row[field].strip() for field in ("prompt", "chosen", "rejected", "rationale")):
            raise ValueError(f"Empty required field: {row['id']}")
        if row["chosen"].strip() == row["rejected"].strip():
            raise ValueError(f"Chosen and rejected responses are identical: {row['id']}")
        if row["condition"] not in REQUIRED_CONDITIONS:
            raise ValueError(f"Unknown evidence condition: {row['id']}")
        seen_ids.add(row["id"])
        seen_prompts.add(row["prompt"])
        conditions.add(row["condition"])

    if conditions != REQUIRED_CONDITIONS:
        raise ValueError(f"Expected all evidence conditions; got {sorted(conditions)}")

    sft_rows = [{
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "completion": [{"role": "assistant", "content": row["chosen"]}],
    } for row in pairs]
    dpo_rows = [{
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    } for row in pairs]
    write_jsonl(TRAIN / "sft.jsonl", sft_rows)
    write_jsonl(TRAIN / "dpo.jsonl", dpo_rows)
    print(f"Prepared {len(pairs)} aligned SFT and DPO examples; evidence conditions: {', '.join(sorted(conditions))}")
    print(f"SFT: {TRAIN / 'sft.jsonl'}")
    print(f"DPO: {TRAIN / 'dpo.jsonl'}")


if __name__ == "__main__":
    main()
