import os, time, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
HF_HOME = Path(__file__).parent / ".cache" / "hf"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
torch.set_num_threads(8)
start = time.time()
print(f"Loading {MODEL_ID} on CPU...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True)
model.to("cpu")
model.config.use_cache = False
lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.train()
print(f"Loaded in {time.time()-start:.1f}s; trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}; total={sum(p.numel() for p in model.parameters()):,}", flush=True)
messages = [
    {"role": "user", "content": "Answer using only the evidence. If a detail is missing, say so.\n\nQuestion: Where was the second office, and what year did it open?\n\nEvidence: [P1] The second office was in Portland. The passage does not state the year."},
    {"role": "assistant", "content": "The second office was in Portland [P1], but the evidence does not give the year."},
]
prompt_messages = messages[:1]
prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
full = tokenizer(full_text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=192)
input_ids = full["input_ids"]
labels = input_ids.clone()
labels[:, :min(prompt_ids.shape[1], labels.shape[1])] = -100
attention_mask = full["attention_mask"]
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
print(f"Running one CPU LoRA update; sequence_length={input_ids.shape[1]}...", flush=True)
t0 = time.time()
optimizer.zero_grad(set_to_none=True)
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
loss.backward()
optimizer.step()
print(json.dumps({"status":"success","device":"cpu","loss":float(loss.detach()),"training_step_seconds":round(time.time()-t0,2),"total_seconds":round(time.time()-start,2),"sequence_length":int(input_ids.shape[1]),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"torch_version":torch.__version__}, indent=2), flush=True)
