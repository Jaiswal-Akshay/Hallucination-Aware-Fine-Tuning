# Grounded test evaluation

## Protocol

The original Qwen2.5-0.5B-Instruct checkpoint, its independently trained SFT adapter, and its independently trained DPO adapter were evaluated on the same four held-out cases in `data/grounded/test.jsonl`. The model checkpoint revision is `7ae557604adf67be50417f59c2c2f167def9a775`. All three arms used the same saved prompts and greedy decoding: `do_sample=false`, `num_beams=1`, and `max_new_tokens=128`. The full prompt, evidence, generated answer, model revision, and decoding settings are preserved in `outputs/grounded-test-evaluation/*_predictions.jsonl` and `run_metadata.json`.

The test set was not used to select checkpoints or tune training parameters. Every generated answer (12 total) was manually annotated claim by claim in `data/grounded/test_annotations.jsonl` using `data/grounded/ANNOTATION_GUIDE.md`. The script `score_grounded_test.py` validates annotation/prediction alignment and calculates the reproducible aggregate file `outputs/grounded-test-evaluation/metrics.json`.

## Results by evidence condition

Rates below include the numerator/denominator where applicable. Unsupported-claim rate counts claims labeled **Contradicted**, **Unsupported**, or **Too unclear to assess**; contradiction rate is also shown separately and overlaps that rate. A model can therefore have a high unsupported-claim rate entirely because it asserted claims against conflicting evidence.

| Condition | Arm | Unsupported claims | Contradicted claims | Full correctness (all cases; among covered answers) | Citation support / claim coverage | Abstention accuracy | Over-refusal | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | Base | 4/11 (36.4%) | 0/11 (0%) | 1/1 (100%); 1/1 (100%) | N/A; 0/11 claims cited | N/A | 0/1 | 1/1 |
| Clean | SFT | 0/6 (0%) | 0/6 (0%) | 1/1 (100%); 1/1 (100%) | N/A; 0/6 claims cited | N/A | 0/1 | 1/1 |
| Clean | DPO | 4/11 (36.4%) | 0/11 (0%) | 1/1 (100%); 1/1 (100%) | N/A; 0/11 claims cited | N/A | 0/1 | 1/1 |
| Incomplete | Base | 6/12 (50%) | 0/12 (0%) | 0/1 (0%); 0/1 (0%) | N/A; 0/12 claims cited | 0/1 (0%) | 0/0 | 1/1 |
| Incomplete | SFT | 1/6 (16.7%) | 0/6 (0%) | 0/1 (0%); 0/1 (0%) | N/A; 0/6 claims cited | 0/1 (0%) | 0/0 | 1/1 |
| Incomplete | DPO | 6/12 (50%) | 0/12 (0%) | 0/1 (0%); 0/1 (0%) | N/A; 0/12 claims cited | 0/1 (0%) | 0/0 | 1/1 |
| Irrelevant | Base | 0/1 (0%) | 0/1 (0%) | 1/1 (100%); N/A | N/A; 0/1 claims cited | 1/1 (100%) | 0/0 | 0/1 |
| Irrelevant | SFT | 0/1 (0%) | 0/1 (0%) | 1/1 (100%); N/A | N/A; 0/1 claims cited | 1/1 (100%) | 0/0 | 0/1 |
| Irrelevant | DPO | 0/1 (0%) | 0/1 (0%) | 1/1 (100%); N/A | N/A; 0/1 claims cited | 1/1 (100%) | 0/0 | 0/1 |
| Contradictory | Base | 4/4 (100%) | 4/4 (100%) | 0/1 (0%); 0/1 (0%) | 2/2 cited claims supported (100%); 2/4 claims cited | 0/1 (0%) | 0/0 | 1/1 |
| Contradictory | SFT | 2/2 (100%) | 2/2 (100%) | 0/1 fully correct (0%); 1/1 partial | 2/2 cited claims supported (100%); 2/2 claims cited | 0/1 (0%) | 0/0 | 1/1 |
| Contradictory | DPO | 4/4 (100%) | 4/4 (100%) | 0/1 (0%); 0/1 (0%) | 2/2 cited claims supported (100%); 2/4 claims cited | 0/1 (0%) | 0/0 | 1/1 |

“Citation support” checks whether the cited passage supports the specific statement attached to that citation. In the contradictory case, the citations support the copied statements as descriptions of what P1 or P2 says; the opposite passage conflicts with each underlying proposition. Base and DPO also make an uncited final conclusion. The 100% local citation-support figure in that condition must not be read as evidence that their answer reconciles the conflict. Citation support is N/A where the model supplied no citations; citation coverage is still zero and is shown explicitly.

Answer correctness is rated at the response level. Clean answers contain the requested protections, although base and DPO add unsupported elaborations. In the incomplete case all arms fail because they assert a year absent from the passage instead of identifying it as missing; base and DPO also invent section mappings. SFT echoes both sides in the contradictory case and is marked partial because it does not explicitly identify the evidence conflict or qualify the conclusion. All arms correctly abstain on irrelevant evidence. Over-refusal is zero on the one clean case for every arm.

## Interpretation

- SFT has fewer unsupported claims than base and DPO on the single clean and incomplete examples. It did not cite its claims on those examples, and it still invented the missing year in the incomplete case.
- DPO's outputs are identical to base on these four cases. This pilot provides no observed test-set evidence that this DPO adapter changed behavior.
- All three arms handle the irrelevant example appropriately, but none handles the contradictory case to the required standard. SFT avoids choosing a side; it still fails to state that the evidence conflicts.
- No arm cited claims on clean or incomplete evidence. Citation behavior therefore remains a clear weakness across this test.

## Limits and next evaluation work

There is only one held-out question per evidence condition, and this corpus is a small pilot. Claim counts are not independent observations, and rates can change sharply with one answer. The contradictory passage is a deliberately constructed counterfactual, not an independent source disagreement. Manual annotation involves judgment; review the per-claim rationales and preserve the annotations with any later corrections. These results are descriptive only and do not establish that SFT or DPO reduces hallucinations generally.

Before drawing research conclusions, expand the source-separated test set with many independently authored questions and documents in each condition, preferably have a second annotator review the claims and citations, and rerun the frozen models with the same prompt and decoding protocol. Keep the expanded test set untouched during training and model selection. In the meantime, retain these outputs as a working pilot record; do not tune against these four answers and then report them as an unbiased test.

## Reproduction

Generate all three arms with the existing local models and adapters:

```powershell
.\.venv\Scripts\python.exe evaluate_grounded_test.py
```

After reviewing or editing the annotations, validate and aggregate them:

```powershell
.\.venv\Scripts\python.exe score_grounded_test.py
```

