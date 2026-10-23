# Data layout

The project stores source-like material under `data/raw/` and trainer-ready or split datasets under `data/processed/`.

- `raw/prototype/source/preference_pairs.jsonl` is the original synthetic prototype's annotation source.
- `processed/prototype/train/` and `processed/prototype/eval/` hold its generated training and held-out data. This historical prototype is separate from the grounded experiment.
- `processed/grounded/` contains source and question split manifests, cases, preference pairs, SFT/DPO data, validation preferences, the annotation guide, and manually reviewed test annotations.

The grounded corpus uses whole-source document splits: Constitution and Articles of Confederation for training, Declaration of Independence for validation, and Bill of Rights for test. See [the grounded data README](processed/grounded/README.md), [annotation guide](processed/grounded/ANNOTATION_GUIDE.md), and [preference review](processed/grounded/PREFERENCE_REVIEW.md). Keep validation and test examples out of training and preference-pair construction.

The old synthetic prototype can be regenerated with:

```powershell
.\.venv\Scripts\python.exe src\prepare_training_data.py
```

The data paths are also recorded in [`configs/data.yaml`](../configs/data.yaml).
