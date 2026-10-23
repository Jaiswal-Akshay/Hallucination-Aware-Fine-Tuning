"""Evaluate base, grounded SFT, and grounded DPO on identical held-out test cases."""
import json
import re
from collections import Counter
from pathlib import Path

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "base" / "Qwen2.5-0.5B-Instruct"
CASES_PATH = ROOT / "data" / "processed" / "grounded" / "cases.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "generations" / "grounded-test"
METADATA_PATH = ROOT / "outputs" / "logs" / "grounded-test" / "run_metadata.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT_TEMPLATE = (
    "Answer using only the supplied evidence. Cite passage IDs for factual claims. "
    "If the evidence is insufficient or conflicting, say so rather than guessing."
)
MAX_NEW_TOKENS = 128
ARMS = {
    "base": None,
    "sft": ROOT / "outputs" / "adapters" / "grounded-sft",
    "dpo": ROOT / "outputs" / "adapters" / "grounded-dpo",
}


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
    cases = [case for case in read_jsonl(CASES_PATH) if case["split"] == "test"]
    conditions = Counter(case["condition"] for case in cases)
    expected_conditions = {"clean", "incomplete", "irrelevant", "contradictory"}
    if len(cases) != 4 or set(conditions) != expected_conditions:
        raise ValueError(f"Expected exactly one held-out test case per condition, got {dict(conditions)}")
    for arm, adapter in ARMS.items():
        if adapter is not None and not adapter.is_dir():
            raise FileNotFoundError(f"Missing {arm} grounded adapter at {adapter}")

    torch.set_num_threads(min(8, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    decoding = {
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": 1,
        "pad_token_id": tokenizer.eos_token_id,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shared_prompts = {case["case_id"]: make_prompt(case) for case in cases}

    for arm, adapter_path in ARMS.items():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to("cpu")
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.eval()
        rows = []
        for index, case in enumerate(cases, start=1):
            prompt = shared_prompts[case["case_id"]]
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
            bracketed_ids = re.findall(r"\[([^\]]+)\]", answer)
            rows.append({
                "arm": arm,
                "case_id": case["case_id"],
                "question_id": case["question_id"],
                "source_document_id": case["source_document_id"],
                "split": case["split"],
                "condition": case["condition"],
                "question": case["question"],
                "evidence": case["passages"],
                "prompt": prompt,
                "generated_answer": answer,
                "bracketed_ids_in_answer": bracketed_ids,
                "expected_answer": case["expected_answer"],
                "answerability": case["answerability"],
                "missing_information": case["missing_information"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "adapter_path": str(adapter_path) if adapter_path else None,
                "decoding": decoding,
            })
            print(f"[{arm} {index}/{len(cases)}] {case['case_id']}: {answer}", flush=True)

        output_path = OUTPUT_DIR / f"{arm}_predictions.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": "cpu",
        "transformers_version": transformers.__version__,
        "prompt_template": PROMPT_TEMPLATE,
        "decoding": decoding,
        "test_cases": len(cases),
        "arms": list(ARMS),
        "condition_counts": dict(conditions),
        "note": "All arms use the same 4 held-out test cases, exact prompt strings, greedy decoding, and 128-token limit. No test results are used for checkpoint or hyperparameter selection.",
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved three arms under {OUTPUT_DIR}; metadata: {METADATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
