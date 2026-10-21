"""Run the original checkpoint on held-out grounded evidence cases (CPU)."""
import json
import re
from collections import Counter
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "base" / "Qwen2.5-0.5B-Instruct"
CASES_PATH = ROOT / "data" / "grounded" / "cases.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "grounded-baseline"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT_TEMPLATE = (
    "Answer using only the supplied evidence. Cite passage IDs for factual claims. "
    "If the evidence is insufficient or conflicting, say so rather than guessing."
)
MAX_NEW_TOKENS = 128


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_prompt(case):
    evidence_lines = []
    for passage in case["passages"]:
        line = f'[{passage["passage_id"]}] {passage["text"]}'
        if passage["provenance"] == "synthetic_counterfactual":
            line += " (Synthetic counterfactual passage)"
        evidence_lines.append(line)
    return (
        f"{PROMPT_TEMPLATE}\n\nQuestion: {case['question']}\n\nEvidence:\n"
        + "\n".join(evidence_lines)
    )


def main():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Expected local checkpoint at {MODEL_PATH}")
    cases = [case for case in read_jsonl(CASES_PATH) if case["split"] in {"validation", "test"}]
    conditions = Counter(case["condition"] for case in cases)
    expected = {"clean", "incomplete", "irrelevant", "contradictory"}
    if set(conditions) != expected:
        raise ValueError(f"Held-out cases must cover all four conditions; got {dict(conditions)}")

    torch.set_num_threads(min(8, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cpu")
    model.eval()

    decoding = {
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": None,
        "top_p": None,
        "num_beams": 1,
        "pad_token_id": tokenizer.eos_token_id,
    }
    rows = []
    for index, case in enumerate(cases, start=1):
        prompt = make_prompt(case)
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=encoded["input_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        supplied_ids = [p["passage_id"] for p in case["passages"]]
        citations = re.findall(r"\[([^\]]+)\]", answer)
        rows.append({
            "case_id": case["case_id"],
            "question_id": case["question_id"],
            "source_document_id": case["source_document_id"],
            "split": case["split"],
            "condition": case["condition"],
            "question": case["question"],
            "evidence": case["passages"],
            "prompt": prompt,
            "generated_answer": answer,
            # These are raw bracketed IDs, not necessarily actual citations. A
            # model may echo the evidence block, so human review is required.
            "bracketed_passage_ids_in_generated_text": citations,
            "all_bracketed_ids_exist_in_case": bool(citations) and all(cid in supplied_ids for cid in citations),
            "answerability": case["answerability"],
            "expected_answer": case["expected_answer"],
            "missing_information": case["missing_information"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "decoding": decoding,
        })
        print(f"[{index}/{len(cases)}] {case['split']}/{case['condition']} {case['case_id']}: {answer}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_path": str(MODEL_PATH),
        "device": "cpu",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "prompt_template": PROMPT_TEMPLATE,
        "decoding": decoding,
        "examples": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "condition_counts": dict(Counter(row["condition"] for row in rows)),
        "outputs_containing_bracketed_passage_ids": sum(bool(row["bracketed_passage_ids_in_generated_text"]) for row in rows),
        "note": "Bracketed IDs may be echoed from the prompt/evidence and are not counted as citations. This run does not automatically score citations or answer correctness; manually annotate claims using data/grounded/ANNOTATION_GUIDE.md.",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} held-out predictions and run metadata under {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
