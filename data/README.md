# Dataset layout

**Important:** The current `source/preference_pairs.jsonl`, `train/*.jsonl`, and `eval/baseline.jsonl` are synthetic pipeline prototypes. The new source-grounded split is separate in `grounded/`; do not combine these corpora. The grounded data has cases and split manifests, but does not yet have preference pairs for training.

- `source/preference_pairs.jsonl` is the annotated authoring source. Each row contains an ID, evidence condition, shared prompt, chosen response, rejected response, and rationale.
- `train/sft.jsonl` is generated in TRL prompt/completion format.
- `train/dpo.jsonl` is generated in TRL prompt/chosen/rejected format.
- `eval/baseline.jsonl` is held out. Do not use it to train or select examples.

Regenerate the two training files with:

```powershell
.\.venv\Scripts\python.exe prepare_training_data.py
```

The current training set is intentionally a tiny prototype (12 pairs, three per evidence condition). Review every source pair for factual support and preference quality before using it. It is not large or diverse enough for a meaningful final model comparison. Expand the training set while keeping the baseline evaluation examples and prompts separate.

The generated training files contain only the columns expected by the selected TRL formats. The authoring source retains rationales for human inspection; rationales are not passed to the trainer.

For the source-grounded document/question split, see [grounded/README.md](grounded/README.md). Keep validation and test cases completely out of training and preference-pair construction.
