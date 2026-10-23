# Repository layout and reproducibility

The project keeps runnable Python entry points in `src/`, versioned run settings in `configs/`, source material and derived datasets in `data/raw/` and `data/processed/`, and local generated artifacts in separate `outputs/` categories. The downloaded model, virtual environments, caches, and trained adapter/prediction artifacts remain local and are ignored by Git.

```text
project/
├── README.md
├── requirements.txt
├── requirements-cpu.txt
├── requirements-lock.txt
├── configs/
│   ├── data.yaml
│   ├── sft.yaml
│   ├── dpo.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/prototype/source/
│   └── processed/
│       ├── grounded/
│       └── prototype/{eval,train}/
├── src/
│   ├── prepare_training_data.py
│   ├── train_grounded_sft.py
│   ├── train_grounded_dpo.py
│   ├── evaluate_grounded_test.py
│   └── score_grounded_test.py
├── outputs/ (ignored local artifacts)
│   ├── adapters/
│   ├── generations/
│   ├── metrics/
│   └── logs/
├── models/ (ignored local checkpoint)
└── docs/
```

`configs/*.yaml` are human-readable records of the project’s effective settings. The Python entry points currently define their runtime settings directly; when changing a config, update the matching script constants and preserve the resulting `run_config.json`, package versions, trainer state, and logs in the adapter output directory. Training scripts resolve paths from the project root, so run them from any working directory with the `src/` path shown below.

## Grounded experiment commands

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe src\train_grounded_sft.py
.\.venv\Scripts\python.exe src\train_grounded_dpo.py
.\.venv\Scripts\python.exe src\evaluate_grounded_test.py
.\.venv\Scripts\python.exe src\score_grounded_test.py
```

The test set remains excluded from training and validation. Predictions, metrics, and run metadata are stored separately under `outputs/generations/grounded-test/`, `outputs/metrics/grounded-test/`, and `outputs/logs/grounded-test/`.

## Data organization

- `data/raw/prototype/source/` contains the earlier synthetic pipeline source, retained for historical comparison.
- `data/processed/grounded/` contains the source-separated grounded pilot corpus, trainer-ready data, annotation guide, and manual test annotations.
- `data/processed/prototype/` contains the earlier synthetic demonstration splits. It is not part of the grounded experiment.
- `outputs/adapters/` holds local adapters plus their effective configs and training logs; `outputs/generations/` holds raw predictions; `outputs/metrics/` holds calculated metrics; `outputs/logs/` holds run metadata and pilot logs.

The adapter files and generated predictions are intentionally Git-ignored because they are reproducible derived artifacts and can be large. Keep their configs, package versions, seeds, and source data in the project. Do not move the base checkpoint into source control.
