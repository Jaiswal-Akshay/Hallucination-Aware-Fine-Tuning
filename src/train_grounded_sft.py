"""Train a CPU LoRA SFT adapter using only grounded training cases."""
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import transformers
import trl
import peft
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "base" / "Qwen2.5-0.5B-Instruct"
TRAIN_PATH = ROOT / "data" / "processed" / "grounded" / "sft.jsonl"
CASES_PATH = ROOT / "data" / "processed" / "grounded" / "cases.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "adapters" / "grounded-sft"
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


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def main():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Expected local checkpoint at {MODEL_PATH}")
    train_rows = read_jsonl(TRAIN_PATH)
    cases = read_jsonl(CASES_PATH)
    train_cases = [case for case in cases if case["split"] == "train"]
    validation_cases = [case for case in cases if case["split"] == "validation"]
    test_cases = [case for case in cases if case["split"] == "test"]
    if len(train_rows) != len(train_cases) or len(validation_cases) == 0:
        raise ValueError("Expected one SFT row per train case and at least one validation case")

    train_by_prompt = {row["prompt"]: row for row in train_rows}
    expected_train_prompts = {make_prompt(case) for case in train_cases}
    if set(train_by_prompt) != expected_train_prompts:
        raise ValueError("Grounded SFT prompts do not match exactly the training-split cases")
    if any(case["split"] != "train" for case in train_cases):
        raise ValueError("Non-training case found in training split")
    validation_prompt_texts = {make_prompt(case) for case in validation_cases}
    validation_rows = [{
        "prompt": [{"role": "user", "content": make_prompt(case)}],
        "completion": [{"role": "assistant", "content": case["expected_answer"]}],
    } for case in validation_cases]
    train_rows_for_trainer = [{
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "completion": [{"role": "assistant", "content": row["completion"]}],
    } for row in train_rows]
    if validation_prompt_texts & set(train_by_prompt):
        raise ValueError("Training/validation prompt leakage detected")

    torch.set_num_threads(min(8, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    config = SFTConfig(
        output_dir=str(OUTPUT_PATH),
        use_cpu=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=1e-4,
        optim="adamw_torch",
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED,
        gradient_checkpointing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train_rows_for_trainer),
        eval_dataset=Dataset.from_list(validation_rows),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    result = trainer.train()
    trainer.save_model(str(OUTPUT_PATH))
    tokenizer.save_pretrained(OUTPUT_PATH)
    trainer.save_state()

    packages = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "datasets": package_version("datasets"),
        "accelerate": package_version("accelerate"),
    }
    run_config = {
        "method": "TRL SFTTrainer with PEFT LoRA",
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
        "base_checkpoint_path": str(MODEL_PATH),
        "initialized_from_original_base": True,
        "training_data": str(TRAIN_PATH),
        "validation_source": str(CASES_PATH),
        "training_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "test_examples_excluded": len(test_cases),
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "lora": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "training_arguments": config.to_dict(),
        "package_versions": packages,
    }
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    (OUTPUT_PATH / "training_logs.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    summary = {
        "training_loss": result.training_loss,
        "train_runtime_seconds": result.metrics.get("train_runtime"),
        "train_steps": trainer.state.global_step,
        "final_validation_metrics": trainer.evaluate(),
        "adapter_path": str(OUTPUT_PATH),
    }
    (OUTPUT_PATH / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
