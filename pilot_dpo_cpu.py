import os, time, json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
os.environ["HF_HOME"] = str(Path(__file__).parent / ".cache" / "hf")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
torch.set_num_threads(8)
start = time.time()
print(f"Loading {MODEL_ID} for one-step DPO CPU pilot...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True)
rows = [
    {
        "prompt": [{"role": "user", "content": "Answer only from the evidence. Cite passage IDs and disclose missing information.\n\nQuestion: Where was the second office, and what year did it open?\n\nEvidence: [P1] The second office was in Portland. The passage does not state the year."}],
        "chosen": [{"role": "assistant", "content": "The second office was in Portland [P1], but the evidence does not give the year."}],
        "rejected": [{"role": "assistant", "content": "The second office was in Portland and opened in 2018 [P1]."}],
    },
    {
        "prompt": [{"role": "user", "content": "Answer only from the evidence. Cite passage IDs and disclose missing information.\n\nQuestion: What is the stated launch month?\n\nEvidence: [P2] The product launched in October."}],
        "chosen": [{"role": "assistant", "content": "The product launched in October [P2]."}],
        "rejected": [{"role": "assistant", "content": "The product launched in October 2021 [P2]."}],
    },
]
config = DPOConfig(
    output_dir="outputs/dpo-pilot",
    use_cpu=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=1,
    learning_rate=1e-5,
    optim="adamw_torch",
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    max_length=160,
    disable_dropout=True,
    precompute_ref_log_probs=True,
    dataloader_num_workers=0,
)
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=Dataset.from_list(rows),
    processing_class=tokenizer,
    peft_config=LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
)
print("Starting one DPO optimizer step on CPU...", flush=True)
t0 = time.time()
result = trainer.train()
print(json.dumps({"status":"success","device":"cpu","total_seconds":round(time.time()-start,2),"training_seconds":round(time.time()-t0,2),"train_loss":result.training_loss,"torch_version":torch.__version__}, indent=2), flush=True)
