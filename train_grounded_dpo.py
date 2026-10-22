"""Train a CPU LoRA DPO adapter on grounded training preferences."""
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import peft
import torch
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "base" / "Qwen2.5-0.5B-Instruct"
TRAIN_PATH = ROOT / "data" / "grounded" / "dpo.jsonl"
SOURCE_PAIRS_PATH = ROOT / "data" / "grounded" / "preference_pairs.jsonl"
VALIDATION_PATH = ROOT / "data" / "grounded" / "dpo_validation.jsonl"
CASES_PATH = ROOT / "data" / "grounded" / "cases.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "grounded-dpo-adapter"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_LENGTH = 256
SEED = 42


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_prompt(case):
    evidence = []
    for passage in case["passages"]:
        line = f'[{passage["passage_id"]}] {passage["text"]}'
        if passage["provenance"] == "synthetic_counterfactual":
            line += " (Synthetic counterfactual passage)"
        evidence.append(line)
    return (
        "Answer using only the supplied evidence. Cite passage IDs for factual claims. "
        "If the evidence is insufficient or conflicting, say so rather than guessing.\n\n"
        f"Question: {case['question']}\n\nEvidence:\n" + "\n".join(evidence)
    )


def conversational(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def main():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Expected local checkpoint at {MODEL_PATH}")
    train_rows = read_jsonl(TRAIN_PATH)
    source_pairs = read_jsonl(SOURCE_PAIRS_PATH)
    validation_rows = read_jsonl(VALIDATION_PATH)
    cases = read_jsonl(CASES_PATH)
    cases_by_id = {case["case_id"]: case for case in cases}
    train_ids = {case["case_id"] for case in cases if case["split"] == "train"}
    validation_ids = {case["case_id"] for case in cases if case["split"] == "validation"}
    test_ids = {case["case_id"] for case in cases if case["split"] == "test"}

    if not train_rows or not validation_rows:
        raise ValueError("DPO requires non-empty training and validation preference data")
    if len(train_rows) != 8 or len(source_pairs) != len(train_rows) or len(validation_rows) != len(validation_ids):
        raise ValueError("Expected all eight training preferences and one validation pair per validation case")
    if any(pair["case_id"] not in train_ids for pair in source_pairs):
        raise ValueError("DPO training file contains a non-training case")
    if any(row["case_id"] not in validation_ids for row in validation_rows):
        raise ValueError("Validation preference file contains a non-validation case")
    if any(pair["case_id"] in test_ids for pair in source_pairs) or any(
        row["case_id"] in test_ids for row in validation_rows
    ):
        raise ValueError("Test cases must remain excluded")
    for row, source in zip(train_rows, source_pairs):
        if any(row[field] != source[field] for field in ("prompt", "chosen", "rejected")):
            raise ValueError(f"DPO data differs from reviewed preference source {source['id']}")
    rows_with_case_ids = [(row, source["case_id"]) for row, source in zip(train_rows, source_pairs)]
    rows_with_case_ids.extend((row, row["case_id"]) for row in validation_rows)
    for row, case_id in rows_with_case_ids:
        case = cases_by_id[case_id]
        if row["prompt"] != make_prompt(case):
            raise ValueError(f"Prompt does not match fixed evidence for {case_id}")
        if row["chosen"].strip() == row["rejected"].strip():
            raise ValueError(f"Chosen and rejected responses are identical for {case_id}")
    for row in validation_rows:
        if row["chosen"] != cases_by_id[row["case_id"]]["expected_answer"]:
            raise ValueError(f"Validation chosen response differs from expected answer for {row['case_id']}")

    torch.set_num_threads(min(8, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    config = DPOConfig(
        output_dir=str(OUTPUT_PATH),
        use_cpu=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=5e-6,
        beta=0.1,
        optim="adamw_torch",
        max_length=MAX_LENGTH,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        disable_dropout=True,
        precompute_ref_log_probs=True,
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED,
        gradient_checkpointing=False,
    )
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list([conversational(row) for row in train_rows]),
        eval_dataset=Dataset.from_list([conversational(row) for row in validation_rows]),
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    package_versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "datasets": package_version("datasets"),
        "accelerate": package_version("accelerate"),
    }
    run_config = {
        "method": "TRL DPOTrainer with PEFT LoRA",
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "base_checkpoint_path": str(MODEL_PATH),
        "initialized_from_original_base": True,
        "initialized_from_sft": False,
        "training_data": str(TRAIN_PATH),
        "validation_data": str(VALIDATION_PATH),
        "training_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "test_examples_excluded": len(test_ids),
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "beta": 0.1,
        "lora": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "training_arguments": config.to_dict(),
        "package_versions": package_versions,
    }
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    result = trainer.train()
    trainer.save_model(str(OUTPUT_PATH))
    tokenizer.save_pretrained(OUTPUT_PATH)
    trainer.save_state()
    logs = trainer.state.log_history
    (OUTPUT_PATH / "training_logs.json").write_text(json.dumps(logs, indent=2, default=str), encoding="utf-8")
    eval_logs = [entry for entry in logs if "eval_loss" in entry]
    summary = {
        "training_loss": result.training_loss,
        "train_runtime_seconds": result.metrics.get("train_runtime"),
        "train_steps": trainer.state.global_step,
        "final_validation_metrics": eval_logs[-1] if eval_logs else None,
        "adapter_path": str(OUTPUT_PATH),
    }
    (OUTPUT_PATH / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
