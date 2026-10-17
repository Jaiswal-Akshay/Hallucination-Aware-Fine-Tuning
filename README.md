# Evidence-Aware Preference Optimization for RAG

This project tests whether evidence-aware preference fine-tuning reduces unsupported claims in retrieval-augmented question answering.

## Environment

The training environment is a project-local Python 3.13 virtual environment, configured for Windows CPU training. PyTorch and torchvision use official CPU-only wheels. The dependency versions are pinned in `requirements-cpu.txt` and `requirements.txt`, with the complete resolved set in `requirements-lock.txt`.

From this folder in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt -r requirements.txt
```

For an exact install of the complete resolved environment, use the lockfile instead:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The CPU wheel index is configured in the requirements files. The PyTorch CPU wheel source follows the official Windows CPU installation guidance: [PyTorch previous-version installation commands](https://pytorch.org/get-started/previous-versions/).

Verify the environment:

```powershell
.\.venv\Scripts\python.exe -c "import torch, transformers, datasets, peft, trl, pandas, sklearn; print('torch', torch.__version__); print('transformers', transformers.__version__); print('datasets', datasets.__version__); print('peft', peft.__version__); print('trl', trl.__version__); print('pandas', pandas.__version__); print('scikit-learn', sklearn.__version__); print('cuda available', torch.cuda.is_available())"
```

Expected device result is `cuda available False`; CPU training is intentional. Do not activate `.venv-pilot`; it is retained only as a backup from the initial feasibility check and is not the clean environment.

## Pilot scripts

```powershell
.\.venv\Scripts\python.exe pilot_cpu.py
.\.venv\Scripts\python.exe pilot_dpo_cpu.py
```

The first checks a single LoRA supervised update. The second checks one TRL DPO optimizer step on synthetic data. These are environment smoke tests, not evidence that the model has learned to avoid hallucinations.

## Data and model storage

The pilot checkpoint cache is stored under `.cache\hf` inside this project. The cache, virtual environments, and generated outputs are ignored by Git. Never put tokens, private corpora, or data without redistribution rights into this repository.

The installed Ollama `llama3:latest` is for local inference reference only. The controlled base/SFT/DPO experiment uses the same Hugging Face checkpoint for all three arms.


## Trainer data formats

For supervised fine-tuning, use prompt/completion examples. For DPO, use explicit `prompt`, `chosen`, and `rejected` fields. See the current [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/en/sft_trainer) and [TRL dataset format guide](https://huggingface.co/docs/trl/en/dataset_formats).

