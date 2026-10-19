# Model checkpoint

## Pinned base model

- Hugging Face repository: `Qwen/Qwen2.5-0.5B-Instruct`
- Resolved revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Project-local path: `models/base/Qwen2.5-0.5B-Instruct`
- Weight format: `model.safetensors` (988,097,824 bytes)
- Weight SHA-256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
- Tokenizer: Qwen2.5 tokenizer files in the same folder (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`)

The checkpoint was already present in the project Hugging Face cache from the CPU pilot. Its snapshot was copied into the stable project-local path above; no new network download was needed. The model directory is ignored by Git because it contains large model weights.

## Experiment initialization rule

All core comparisons use this exact local checkpoint and tokenizer:

1. **Base:** load the original checkpoint without fine-tuning.
2. **SFT:** initialize from the original checkpoint, then train the SFT adapter.
3. **DPO:** initialize separately from the original checkpoint, then train the DPO adapter. Do not start DPO from SFT.

Keep `llama3:latest` in Ollama as an optional, separately labeled inference reference. It is a different model family and is not part of the core base/SFT/DPO comparison.

## Loading locally

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/base/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    dtype="float32",
)
```

The pilot scripts already use this path. Keep future training and evaluation scripts pointed at it so all arms share the same pinned base files.
