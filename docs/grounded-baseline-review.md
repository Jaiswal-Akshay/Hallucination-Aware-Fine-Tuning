# Grounded Qwen baseline review

## Run

Before any grounded fine-tuning, the original local `Qwen/Qwen2.5-0.5B-Instruct` checkpoint was run on all held-out validation and test cases: eight examples total, two per evidence condition. The same evidence-only instruction used by the grounded SFT/DPO examples was used for every case and model arm. No adapter was loaded. The run was CPU-only and deterministic (`do_sample=false`, one beam, maximum 128 new tokens).

Run it again from the project root with:

```powershell
.\.venv\Scripts\python.exe evaluate_grounded_baseline.py
```

The script saves the full prompt, evidence passages, raw generated answer, case metadata, model revision, and decoding settings under the ignored local directory `outputs/grounded-baseline/`. `predictions.jsonl` contains one complete record per case; `run_metadata.json` records the shared setup. The pinned model revision is `7ae557604adf67be50417f59c2c2f167def9a775`.

## Manual format and behavior review

| Condition | Baseline observation |
| --- | --- |
| Clean | Answered the Declaration date and listed the First Amendment protections, but neither answer cited a passage ID. The clean Bill of Rights response also added explanatory claims beyond the passage. |
| Incomplete | Did not mark requested missing information. For the Declaration case it asserted a city absent from the passage. For the Bill of Rights case it added amendment/article mappings and a ratification date absent from the supplied passage. |
| Irrelevant | The Bill of Rights population question was appropriately declined. The Declaration/coinage question instead produced unsupported claims and an invalid inference from unrelated evidence. |
| Contradictory | In both cases, it failed to acknowledge the explicit conflict and confidently chose one side. In the Bill of Rights case it echoed the evidence block before choosing P1. |

The citation-format requirement was **not demonstrated**: no response attached a passage ID to its answer claims. The lone appearance of `[P1]` and `[P2]` in one generated answer was copied evidence text, not claim citations. This is why the output metadata treats bracketed IDs as raw text only and does not score citation support automatically.

This run confirms that the model can load and generate on the chosen prompt format, but it does **not** reliably follow the grounded answer/citation behavior. It is a tiny diagnostic baseline, not a statistically meaningful estimate of hallucination rate. Use the claim-level annotation guide for manual scoring and keep the test cases held out from training.
