# Grounded DPO CPU pilot

## Run configuration

- **Base:** `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`
- **Method:** TRL `DPOTrainer` with PEFT LoRA; the DPO adapter was initialized independently from the original base, not from SFT
- **Training data:** 8 reviewed preference pairs from `data/processed/grounded/dpo.jsonl` (training split only)
- **Validation data:** 4 validation-only preference pairs in `data/processed/grounded/dpo_validation.jsonl`, one per evidence condition; each chosen response matches that case's expected answer
- **Test data:** 4 test cases excluded
- **Sequence cap:** 256 tokens
- **LoRA:** rank 4, alpha 8, dropout 0.05, `q_proj` and `v_proj`
- **Batch size:** 1; gradient accumulation: 1; epochs: 1; learning rate: `5e-6`; DPO beta: `0.1`
- **Seed:** 42 (also used as `data_seed`)
- **Device:** CPU; float32 model weights, trainer `bf16=true` and `fp16=false`; reference log probabilities precomputed
- **Runtime:** about 20 minutes 31 seconds for optimization; validation evaluation about 38 seconds, plus setup and reference-score precomputation

The string-form source preference data was converted internally to TRL's conversational explicit-prompt format so the Qwen chat template is applied consistently. TRL supports both standard explicit `prompt`/`chosen`/`rejected` records and conversational records, and documents the explicit prompt form as recommended: [TRL DPOTrainer dataset formats](https://huggingface.co/docs/trl/en/dpo_trainer).

## Result

- Training steps: 8
- Training loss: `0.6994`
- Validation loss: `0.6830`
- Validation reward accuracy: `1.0` across four pairs
- Validation reward margin: `0.0204`

These are optimization diagnostics on a tiny pilot, not measurements of hallucination reduction. The validation set has only four manually authored pairs, and the test cases remain untouched for the final comparison.

## Saved outputs

The generated files are local and Git-ignored under `outputs/adapters/grounded-dpo/`:

- `adapter_model.safetensors` and `adapter_config.json`: trained LoRA adapter
- `run_config.json`: base revision, split counts, seed, DPO/LoRA settings, full trainer arguments, and package versions
- `training_logs.json` and `trainer_state.json`: step and validation logs
- `training_summary.json`: final train and validation metrics
- tokenizer files and `chat_template.jinja`

To reproduce from the project root:

```powershell
.\.venv\Scripts\python.exe src\train_grounded_dpo.py
```
