# Step 1: Laptop capacity and training pilot

**Date checked:** 2026-10-17  
**Purpose:** Confirm whether the current laptop can load Qwen2.5-0.5B-Instruct and complete a small CPU fine-tuning update before building the full dataset.

## Hardware inventory

| Item | Observed |
|---|---|
| CPU | Intel Core Ultra 9 185H |
| Physical cores / logical processors | 16 / 22 |
| Installed RAM | 31.43 GiB reported (marketed as 32 GB) |
| Available RAM at initial check | 12.17 GiB |
| Available RAM after initial pilot | 11.09 GiB |
| GPU | Intel Arc integrated graphics, approximately 2 GiB reported adapter memory |
| CUDA | Not available to PyTorch |
| OS | Windows 11 Home Single Language, build 26300 |
| Python | 3.13.3, 64-bit |
| Free disk | D: 507.86 GB; C: 59.48 GB |

The Intel Arc is not being used by this training environment. The project venv uses CPU-only PyTorch and torchvision. The installed Ollama `llama3:latest` model (4.7 GB) remains an inference-only reference; it is not the training checkpoint for this experiment.

## Project environment

The project-local `.venv` uses the existing Python installation as a system package source, with training components installed in the project environment. The relevant versions observed during the pilot were:

- PyTorch `2.9.0+cpu`
- Transformers `5.16.1`
- PEFT `0.21.2`
- TRL `1.16.0`
- Datasets `5.1.0`
- Python `3.13.3`

PyTorch and torchvision were installed as matching CPU-only builds. This was necessary because the starting global PyTorch build was `2.9.0+cu130` and crashed during CPU forward execution while Transformers queried CUDA stream state. The matching CPU build fixed that runtime crash. These installs are scoped to this project's `.venv`; `.venv` and the Hugging Face model cache are ignored by Git.

## Model load and SFT-style LoRA smoke test

- Checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`
- Parameters loaded: 494,167,936
- Adapter parameters trained: 135,168
- Device: CPU, float32
- Sequence length: 101 tokens
- Training: one forward, backward, and AdamW adapter update
- Result: **passed**
- Update time: **1.47 seconds**
- Loss: 1.32165

This verifies the model can load and a short adapter-training step can complete on the laptop.

## Actual DPO smoke test

A separate TRL DPO test used two synthetic preference pairs, one CPU optimizer step, LoRA rank 2, batch size 1, maximum sequence length 160, and precomputed reference log probabilities.

- Result: **passed**
- DPO optimizer-step runtime: **134.7 seconds**
- Total runtime including model startup, tokenization, and reference-log-probability precomputation: **151.6 seconds**
- Training loss: 0.69315
- Sampled process working set during execution: approximately 2.65 GiB (not a peak-memory measurement)
- No out-of-memory error occurred.

The first DPO attempt used plain string prompts and produced a token-boundary warning. The pilot was changed to the model's conversational format and rerun successfully without that warning. The successful script is `pilot_dpo_cpu.py`.

One optimizer step is only a software/hardware feasibility check. The loss and two-example dataset do not demonstrate that DPO has learned the desired behavior.

## Feasibility decision

**Decision: proceed with a small CPU pilot using Qwen2.5-0.5B-Instruct and LoRA.** The laptop can execute the core DPO training path without a GPU. However, the measured DPO rate is slow: about 2.25 minutes per batch-size-1 optimizer step on these short examples. At that observed rate, 100 steps would take about 3.75 hours and 300 steps about 11.2 hours, before accounting for longer contexts, validation, checkpoints, and evaluation. These are rough linear estimates, not runtime guarantees.

Recommended next scale:

1. Start with 20–50 preference pairs and short contexts.
2. Keep batch size 1 initially; measure a complete short epoch and actual memory peak.
3. Use one epoch for the first end-to-end pilot, with a small held-out set.
4. Scale toward 100–300 pairs only if wall time remains acceptable.
5. If the research requires a larger dataset or several epochs, move training to a GPU environment while keeping the dataset and evaluation protocol fixed.

We still need to measure peak memory during a full dataset run. Available RAM varies as other applications open, so close memory-heavy applications before training.

## Files created for this step

- `pilot_cpu.py` — one-step LoRA/SFT-style CPU training smoke test.
- `pilot_dpo_cpu.py` — one-step TRL DPO CPU smoke test.
- `.venv/` — ignored project-local Python environment.
- `.cache/hf/` — ignored local checkpoint cache.
- `requirements.txt` — pinned Python training dependencies; install CPU PyTorch separately as noted in the project setup instructions.

## Re-run the smoke tests

From this project directory, activate the environment or invoke it directly:

```powershell
.\.venv\Scripts\python.exe pilot_cpu.py
.\.venv\Scripts\python.exe pilot_dpo_cpu.py
```

For a fresh environment, install the Windows CPU wheels first, then the pinned packages in `requirements.txt`:

```powershell
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The official PyTorch previous-version page lists the matching 2.9.0 CPU-only wheel command for Windows. See [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).

## Step 2: isolated environment

The original pilot environment inherited packages from the global Python installation. To satisfy project isolation, it has been preserved at `.venv-pilot/` and must not be used for new work. The clean `.venv/` is created without `--system-site-packages` and receives all runtime dependencies from the pinned files:

- `requirements-cpu.txt` pins CPU-only PyTorch `2.9.0+cpu` and torchvision `0.24.0+cpu` from the official PyTorch CPU index.
- `requirements.txt` pins Transformers, Datasets, PEFT, TRL, pandas, scikit-learn, and supporting packages.

The existing Python 3.13 installation provides only the interpreter; runtime packages are installed into `.venv/`. The exact install and verification commands are in `README.md`.
