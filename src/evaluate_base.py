"""Run the pinned base checkpoint on the small held-out RAG evaluation set."""
import json
import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "base" / "Qwen2.5-0.5B-Instruct"
DATA_PATH = ROOT / "data" / "processed" / "prototype" / "eval" / "baseline.jsonl"
MAX_NEW_TOKENS = 96


def load_examples():
    with DATA_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("base", "sft", "dpo"), default="base")
    arm = parser.parse_args().arm
    run_name = "baseline" if arm == "base" else f"{arm}-prototype"
    generation_dir = ROOT / "outputs" / "generations" / run_name
    metrics_path = ROOT / "outputs" / "metrics" / run_name / "metrics.json"
    torch.set_num_threads(min(8, torch.get_num_threads()))
    examples = load_examples()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cpu")
    if arm in ("sft", "dpo"):
        adapter_path = ROOT / "outputs" / "adapters" / f"{arm}-prototype"
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()

    rows = []
    for i, ex in enumerate(examples, start=1):
        messages = [{"role": "user", "content": ex["prompt"]}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )["input_ids"]
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        found_citations = sorted(set(re.findall(r"\[([A-Za-z]+\d+)\]", answer)))
        # These are transparent screening heuristics, not a semantic hallucination judge.
        discloses_uncertainty = bool(re.search(
            r"\b(does not state|doesn't state|not provided|cannot determine|can't determine|unknown|unclear|conflict|conflicting|unable to (?:answer|determine)|insufficient information|not mention|not identify|not specify)\b",
            answer, flags=re.IGNORECASE,
        ))
        answer_lower = answer.lower()
        key_terms_present = [term for term in ex["answer_key"] if term.lower() in answer_lower]
        rows.append({
            "id": ex["id"], "condition": ex["condition"], "prompt": ex["prompt"],
            "prediction": answer, "reference": ex["reference"],
            "should_abstain": ex["should_abstain"],
            "expected_citations": ex["expected_citations"],
            "found_citations": found_citations,
            "citation_ids_in_expected_set": all(c in ex["expected_citations"] for c in found_citations),
            "all_expected_citations_present": all(c in found_citations for c in ex["expected_citations"]),
            "discloses_uncertainty_heuristic": discloses_uncertainty,
            "abstention_heuristic_matches": discloses_uncertainty == ex["should_abstain"],
            "answer_key_terms_present": key_terms_present,
            "answer_key_terms": ex["answer_key"],
            "notes": ex["notes"],
        })
        print(f"[{i}/{len(examples)}] {ex['id']}: {answer}", flush=True)

    generation_dir.mkdir(parents=True, exist_ok=True)
    with (generation_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    metrics = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "arm": arm,
        "checkpoint_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "evaluation_examples": len(rows),
        "metric_note": "Heuristic screening only. Review predictions manually; citation presence and keyword overlap do not establish factual support or correctness.",
        "by_condition": {},
    }
    for condition, group in sorted(grouped.items()):
        citation_examples = [r for r in group if r["expected_citations"]]
        expected_citation_count = sum(len(r["expected_citations"]) for r in citation_examples)
        expected_citations_found = sum(
            sum(c in r["found_citations"] for c in r["expected_citations"])
            for r in citation_examples
        )
        metrics["by_condition"][condition] = {
            "n": len(group),
            "expected_citations_found": expected_citation_count and expected_citations_found,
            "expected_citation_count": expected_citation_count,
            "answer_key_term_recall_proxy": round(
                sum(len(r["answer_key_terms_present"]) / max(1, len(r["answer_key_terms"])) for r in group) / len(group), 3
            ),
        }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved predictions under {generation_dir} and proxy metrics under {metrics_path.parent}", flush=True)


if __name__ == "__main__":
    main()
