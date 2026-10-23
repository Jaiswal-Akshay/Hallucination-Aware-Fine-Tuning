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
.\.venv\Scripts\python.exe src\pilot_cpu.py
.\.venv\Scripts\python.exe src\pilot_dpo_cpu.py
```

The first checks a single LoRA supervised update. The second checks one TRL DPO optimizer step on synthetic data. These are environment smoke tests, not evidence that the model has learned to avoid hallucinations.

## Base-model evaluation

The small held-out starter set is `data/processed/prototype/eval/baseline.jsonl` (12 examples across clean, incomplete, irrelevant, and contradictory evidence). Keep this evaluation set out of all training data. Run the original checkpoint before SFT or DPO:

```powershell
.\.venv\Scripts\python.exe src\evaluate_base.py
```

Predictions and simple proxy metrics are written to `outputs/generations/baseline/` and `outputs/metrics/baseline/`. See [docs/baseline-review.md](docs/baseline-review.md) for the manual review. Citation coverage and answer-key overlap do not establish semantic correctness or claim-level support. Expand this starter set before treating results as research evidence.

## Training data prototype

The preference-pair authoring source is `data/raw/prototype/source/preference_pairs.jsonl`. Generate the TRL-compatible SFT and DPO files with:

```powershell
.\.venv\Scripts\python.exe src\prepare_training_data.py
```

This creates `data/processed/prototype/train/sft.jsonl` (`prompt`, `completion`) and `data/processed/prototype/train/dpo.jsonl` (`prompt`, `chosen`, `rejected`). Both training arms use the same prompts and chosen answers; rejected answers are plausible unsupported, contradictory, or overconfident alternatives. The current set has only 12 synthetic examples. Review it and expand it before interpreting any training result. Keep `data/processed/prototype/eval/baseline.jsonl` held out. See [data/README.md](data/README.md).

## Adapter training and comparison

The CPU LoRA training commands each load the original base model independently:

```powershell
.\.venv\Scripts\python.exe src\train_sft.py
.\.venv\Scripts\python.exe src\train_dpo.py
```

They save adapters separately under `outputs/adapters/sft-prototype` and `outputs/adapters/dpo-prototype`. To evaluate each arm on the same held-out examples:

```powershell
.\.venv\Scripts\python.exe src\evaluate_base.py --arm base
.\.venv\Scripts\python.exe src\evaluate_base.py --arm sft
.\.venv\Scripts\python.exe src\evaluate_base.py --arm dpo
```

Predictions are saved under `outputs/generations/` and proxy metrics under `outputs/metrics/`. CPU training is slow; the 12-example run is only a pipeline demonstration. Metrics are proxies, and predictions need manual review.

The completed CPU prototype run and its limitations are summarized in [docs/training-results.md](docs/training-results.md).

## Source-grounded evidence split

The grounded corpus uses National Archives primary-source transcriptions about U.S. founding documents. Whole documents and question IDs were assigned to train, validation, and test before passage cases were authored. See [data/processed/grounded/README.md](data/processed/grounded/README.md), the [claim and citation annotation guide](data/processed/grounded/ANNOTATION_GUIDE.md), and the [grounded preference-pair review](data/processed/grounded/PREFERENCE_REVIEW.md). Starter SFT and DPO files use only train cases and are separate from the earlier synthetic prototype. The original Qwen checkpoint has been run on all eight held-out validation/test cases; see the [grounded baseline review](docs/grounded-baseline-review.md). The grounded SFT and DPO training entry points are `src/train_grounded_sft.py` and `src/train_grounded_dpo.py`. The legacy trainers target the synthetic prototype under `data/processed/prototype/train/`.

The grounded CPU SFT and DPO pilots are complete. Their adapters, seeds, full configs, package versions, logs, and metrics are saved under ignored `outputs/adapters/grounded-sft/` and `outputs/adapters/grounded-dpo/`; see [grounded SFT results](docs/grounded-sft-training.md) and [grounded DPO results](docs/grounded-dpo-training.md). The adapters are separate, and the SFT and DPO runs each start from the original base checkpoint. These tiny-pilot metrics do not demonstrate a reduction in hallucinations.

The first three-arm held-out test evaluation and paired results report are complete. The same four test prompts were run with base, SFT, and DPO using greedy decoding and identical generation settings; all 12 answers were manually annotated claim by claim. See the [consolidated results report](docs/grounded-results-report.md), [condition-level evaluation report](docs/grounded-test-evaluation.md), annotation records at `data/processed/grounded/test_annotations.jsonl`, and the reproducible scorer `src/score_grounded_test.py`. The results are descriptive only: one test question per evidence condition is far too small to support a general claim about hallucination reduction.

For a narrated walkthrough of the research simulation, dataset, trust safeguards, and current limitations, see [docs/video-recording-script.md](docs/video-recording-script.md).

## Data and model storage

Python entry points are in `src/`; current run settings are recorded in `configs/`. Source-like prototype material is in `data/raw/`, while the grounded dataset and trainer-ready splits are under `data/processed/`. Local outputs are divided into `outputs/adapters/`, `outputs/generations/`, `outputs/metrics/`, and `outputs/logs/`. See the [repository layout and reproducibility notes](docs/repository-layout.md).

The pinned Qwen checkpoint and tokenizer are stored at `models/base/Qwen2.5-0.5B-Instruct`; its Hub revision and files are recorded in [docs/model-checkpoint.md](docs/model-checkpoint.md). Both pilot scripts load this local folder with network access disabled. Use this same original checkpoint independently for base evaluation, SFT, and DPO; do not initialize DPO from SFT. The model files, Hugging Face cache, virtual environments, and generated outputs are ignored by Git. Never put tokens, private corpora, or data without redistribution rights into this repository.

The installed Ollama `llama3:latest` is for local inference reference only. The controlled base/SFT/DPO experiment uses the same Hugging Face checkpoint for all three arms.


## Trainer data formats

For supervised fine-tuning, use prompt/completion examples. For DPO, use explicit `prompt`, `chosen`, and `rejected` fields. See the current [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/en/sft_trainer) and [TRL dataset format guide](https://huggingface.co/docs/trl/en/dataset_formats).

