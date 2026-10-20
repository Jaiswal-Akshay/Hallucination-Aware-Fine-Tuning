"""Train the SFT LoRA adapter from the original pinned base checkpoint."""
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "base" / "Qwen2.5-0.5B-Instruct"
DATA_PATH = ROOT / "data" / "train" / "sft.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "sft-adapter"
MAX_LENGTH = 256


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    torch.set_num_threads(8)
    rows = read_jsonl(DATA_PATH)
    if not rows:
        raise ValueError(f"No SFT rows found at {DATA_PATH}; run prepare_training_data.py first")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    config = SFTConfig(
        output_dir=str(OUTPUT_PATH),
        use_cpu=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=1e-4,
        optim="adamw_torch",
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        seed=42,
        gradient_checkpointing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    result = trainer.train()
    trainer.save_model(str(OUTPUT_PATH))
    tokenizer.save_pretrained(OUTPUT_PATH)
    summary = {
        "method": "SFT LoRA",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "base_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "initialized_from_original_base": True,
        "training_examples": len(rows),
        "epochs": 1,
        "max_length": MAX_LENGTH,
        "train_loss": result.training_loss,
        "train_runtime_seconds": result.metrics.get("train_runtime"),
        "adapter_path": str(OUTPUT_PATH),
    }
    (OUTPUT_PATH / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
