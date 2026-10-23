# Hallucination-aware fine-tuning: pilot results

## Executive summary

This report compares the original Qwen2.5-0.5B-Instruct checkpoint with separately trained SFT and DPO LoRA adapters. Each arm answered the same four held-out questions (one clean, incomplete, irrelevant, and contradictory case) with the same prompt and greedy decoding settings. All 12 answers were manually audited.

SFT produced fewer unsupported claim occurrences in this tiny set than the base model, while DPO's text outputs were identical to base on all four cases. SFT still failed to mark missing information, cite answers to clean/incomplete evidence, and explicitly flag conflicting evidence. These observations are preliminary: four test questions cannot support a general claim about hallucination reduction or statistical significance.

## Evaluation set and audit

- **Questions:** 4 held-out questions, each evaluated by all 3 arms.
- **Answers scored:** 12 total (4 per arm).
- **Conditions:** one clean, one incomplete, one irrelevant, and one contradictory question.
- **Manual audit:** all 12 answers were reviewed. The audit annotated 71 factual claim occurrences and 6 citation-to-claim pairs. See [`test_annotations.jsonl`](../data/processed/grounded/test_annotations.jsonl) for claim labels, citation judgments, rationale, answer correctness, abstention, and coverage.
- **Paired design:** each model received the same question, evidence, prompt, and decoding configuration. Results are paired by question; counts are shown instead of significance tests.

### Overall metrics

Unsupported-claim rate counts claims labeled contradicted, unsupported, or too unclear to assess. Contradiction rate is separately reported and overlaps this numerator. “Unsupported only” excludes contradicted claims. These are pooled claim-occurrence rates, not averages over independent questions.

| Metric | Base | SFT | DPO |
|---|---:|---:|---:|
| Questions / answers scored | 4 / 4 | 4 / 4 | 4 / 4 |
| Factual claim occurrences annotated | 28 | 15 | 28 |
| Supported claims | 14/28 (50.0%) | 12/15 (80.0%) | 14/28 (50.0%) |
| Unsupported-claim rate | 14/28 (50.0%) | 3/15 (20.0%) | 14/28 (50.0%) |
| Unsupported-only rate | 10/28 (35.7%) | 1/15 (6.7%) | 10/28 (35.7%) |
| Contradiction rate | 4/28 (14.3%) | 2/15 (13.3%) | 4/28 (14.3%) |
| Answers with ≥1 contradicted/unsupported/unclear claim | 3/4 (75%) | 2/4 (50%) | 3/4 (75%) |
| Fully correct answers, all cases | 2/4 (50%) | 2/4 (50%) | 2/4 (50%) |
| Partial answers | 0/4 | 1/4 (25%) | 0/4 |
| Fully correct among covered answers | 1/3 (33.3%) | 1/3 (33.3%) | 1/3 (33.3%) |
| Coverage | 3/4 (75%) | 3/4 (75%) | 3/4 (75%) |
| Citation claim coverage | 2/28 (7.1%) | 2/15 (13.3%) | 2/28 (7.1%) |
| Citation support among attached citation-claim pairs | 2/2 (100%) | 2/2 (100%) | 2/2 (100%) |
| Abstention accuracy on applicable cases | 1/3 (33.3%) | 1/3 (33.3%) | 1/3 (33.3%) |
| Over-refusal on clean evidence | 0/1 | 0/1 | 0/1 |

The apparent 100% citation support rate has a very small denominator: every model cited only the two echoed statements in the contradictory case. These local citations accurately point to the passage being repeated, but do not resolve the disagreement between P1 and P2. No claims were cited in the clean or incomplete cases.

## Results by evidence condition

| Condition (1 question) | Base | SFT | DPO |
|---|---|---|---|
| Clean | 4/11 claims unsupported (36.4%); fully correct answer with unsupported elaboration; no citations | 0/6 unsupported; fully correct; no citations | Same output and result as base |
| Incomplete | 6/12 unsupported (50%); wrong section mappings and year; fails to mark missing year | 1/6 unsupported (16.7%); still invents a year and fails to mark it missing | Same output and result as base |
| Irrelevant | Correctly abstains; zero unsupported claims | Correctly abstains; zero unsupported claims | Correctly abstains; zero unsupported claims |
| Contradictory | 4/4 claims contradicted; cites both passages, then confidently picks one side | 2/2 claims contradicted; echoes both sides without explicitly saying the evidence conflicts; partial answer | Same output and result as base |

The full metric breakdown, including per-condition correctness, citation counts, coverage, abstention accuracy, and over-refusal, is in the [test evaluation report](grounded-test-evaluation.md) and machine-readable [`metrics.json`](../outputs/metrics/grounded-test/metrics.json).

## Paired comparisons

All comparisons below use the four matched questions. With `n=4`, one question changes an overall question-level rate by 25 percentage points; within a condition, the one observed answer cannot estimate a condition-level rate. The exact counts and denominators are shown because uncertainty is substantial. Claim-level percentages should not be treated as independent-sample estimates because multiple claims come from the same answer. No significance test is warranted.

- **DPO vs base:** generated text was identical on all 4/4 questions. Every answer-level and claim-level result is therefore tied in this run.
- **SFT vs base, clean:** SFT removed the four unsupported elaborations in the base answer while retaining the five requested protection categories. Both answers were rated fully correct; neither cited claims.
- **SFT vs base, incomplete:** the unsupported claim rate fell from 6/12 to 1/6. Both answers remained incorrect because SFT also asserted an unsupported ratification year instead of saying the passage omitted it.
- **SFT vs base, irrelevant:** both produced the same appropriate abstention.
- **SFT vs base, contradictory:** SFT improved from an incorrect confident selection to a partial answer that repeats both sides. It still did not explicitly identify the conflict, so strict abstention accuracy remained false for both.
- **Question-level unsupported-answer incidence:** at least one contradicted, unsupported, or unclear claim appeared in 3/4 base answers, 2/4 SFT answers, and 3/4 DPO answers. This is descriptive and based on four cases only.

## Human audit findings and representative examples

All three model outputs on all four cases were manually reviewed, rather than sampling a subset. Each factual claim was classified against the supplied evidence, citations were checked against their attached claim, and answer correctness, abstention, over-refusal, and coverage were recorded. The annotations include per-claim rationales. There was one annotator, so no inter-annotator agreement or adjudication statistic is available.

**Success — irrelevant evidence:** all arms answered that the passage did not provide the requested New York City population and declined to guess. This is one correct abstention per arm.

**Success — SFT on the clean case:** SFT listed the five protections in the passage without the four unsupported explanations present in the base and DPO outputs. It still omitted the requested passage citation.

**Failure — incomplete evidence:** base and DPO invented Article I section numbers and a ratification date; SFT supplied the year 1791. The passage only listed the protections. All three should have stated that it did not provide the year.

**Failure — contradictory evidence:** base and DPO repeated both P1 and P2, then selected P1 confidently. SFT repeated both passages but did not say they conflict. None met the requested conflict-aware behavior.

## Training time and resource use

| Arm | Training configuration | Optimization time | Validation time | Key training diagnostics |
|---|---|---:|---:|---|
| Base | No fine-tuning | N/A | N/A | Original checkpoint used directly |
| SFT | CPU, 8 training examples, 4 validation examples, 8 steps, 1 epoch, batch 1, max length 256, LoRA rank 4 | 1,519.3 s (25m 19s) | 34.9 s | Train loss 2.0003; validation loss 2.2347 |
| DPO | CPU, 8 preference pairs, 4 validation pairs, 8 steps, 1 epoch, batch 1, max length 256, LoRA rank 4, beta 0.1 | 1,230.7 s (20m 31s) | 38.4 s, plus setup/reference-score precomputation | Train loss 0.6994; validation loss 0.6830; validation reward accuracy 1.0 on 4 pairs |

Both adapters were trained independently from the original base checkpoint with CPU PyTorch 2.9.0+cpu and float32 model weights. The training configs set `bf16=true`; the model weights were loaded in float32. Peak RAM, CPU utilization, and wall-clock time for setup/reference precomputation were not captured, so they are not estimated here. The reported training times cover optimization; they do not represent full end-to-end time. Full configs, logs, and package versions remain in the ignored adapter output folders, summarized in the [SFT training record](grounded-sft-training.md) and [DPO training record](grounded-dpo-training.md).

## Limitations

1. **Test size:** one question in each condition is far too small for stable performance estimates. Each condition-specific result is a single answer, not a population rate.
2. **Small model:** Qwen2.5-0.5B-Instruct is a 0.5-billion-parameter model; behavior may not transfer to larger models or other model families.
3. **Training data:** each adapter saw only eight training examples and four validation examples. The authored examples and narrow founding-document topic do not represent broad RAG usage.
4. **Synthetic conflict:** the contradictory case includes a constructed counterfactual passage, not a naturally occurring disagreement between independent sources.
5. **Annotation:** all outputs were manually judged by one annotator. Claim segmentation and support judgments can be subjective; a second annotator is needed for agreement and adjudication.
6. **Citation behavior:** citation coverage was zero on clean and incomplete evidence. The contradictory-case citation support result measures passage-to-quoted-claim alignment, not successful conflict resolution.
7. **CPU/resource measurement:** CPU training was slow and peak memory/CPU usage were not recorded. The project demonstrates a pipeline pilot, not an efficient or production-scale training recipe.
8. **No significance claim:** paired evaluation controls the question set, but four pairs do not provide enough evidence for a statistical significance test.

## Reproducibility and files

- Test generation: `src/evaluate_grounded_test.py`
- Manual claim annotations: `data/processed/grounded/test_annotations.jsonl`
- Metric aggregation: `src/score_grounded_test.py`
- Raw predictions: `outputs/generations/grounded-test/`
- Run metadata: `outputs/logs/grounded-test/run_metadata.json`
- Machine-readable metrics: `outputs/metrics/grounded-test/metrics.json`
- Condition-level analysis: [`grounded-test-evaluation.md`](grounded-test-evaluation.md)

The next research step is to enlarge the document-separated corpus and held-out test set, add a second annotator, and evaluate frozen models on the expanded set without tuning against it.
