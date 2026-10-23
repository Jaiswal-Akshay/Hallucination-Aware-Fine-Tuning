# Grounded SFT CPU pilot

## Run configuration

- **Base:** `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`
- **Method:** TRL `SFTTrainer` with PEFT LoRA
- **Training data:** 8 chosen responses from `data/processed/grounded/sft.jsonl` (training split only)
- **Validation data:** 4 expected responses built from validation cases in `data/processed/grounded/cases.jsonl`
- **Test data:** 4 test cases excluded
- **Sequence cap:** 256 tokens
- **LoRA:** rank 4, alpha 8, dropout 0.05, `q_proj` and `v_proj`
- **Batch size:** 1; gradient accumulation: 1; epochs: 1; learning rate: `1e-4`
- **Seed:** 42 (also used as `data_seed`)
- **Device:** CPU, float32 model weights
- **Trainer precision setting:** serialized `bf16=true`; the model checkpoint itself was loaded with float32 weights
- **Runtime:** about 25 minutes 19 seconds for training; validation evaluation about 35 seconds

The trainer uses conversational prompt/completion records internally so prompt-only tokenization matches the prompt prefix in the full training sequence. The source SFT file remains the plain `prompt`/`completion` JSONL, aligned with the DPO data.

## Result

- Training steps: 8
- Training loss: `2.0003`
- Validation loss: `2.2347`
- Validation mean token accuracy: `0.6765`

These losses are a pipeline record, not evidence of improved factual grounding. This is a tiny pilot with only eight training cases and four validation cases. Do not use the held-out test cases for tuning.

## Saved outputs

The generated files are local and Git-ignored under `outputs/adapters/grounded-sft/`:

- `adapter_model.safetensors` and `adapter_config.json`: trained LoRA adapter
- `run_config.json`: base revision, dataset split counts, seed, training arguments, LoRA settings, and package versions
- `training_logs.json` and `trainer_state.json`: step and evaluation logs
- `training_summary.json`: final train and validation metrics
- tokenizer files and `chat_template.jinja`

To reproduce from the project root:

```powershell
.\.venv\Scripts\python.exe src\train_grounded_sft.py
```
