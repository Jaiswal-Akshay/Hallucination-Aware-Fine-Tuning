# Grounded evidence cases: U.S. founding documents

## Topic and sources

The starter corpus is narrowly scoped to four primary U.S. founding documents: the Constitution, Articles of Confederation, Declaration of Independence, and Bill of Rights. Each source record, URL, date, access date, and split is in `source_manifest.jsonl`.

The passages in `cases.jsonl` are short paraphrases of facts from those National Archives transcriptions. No archival images or long verbatim excerpts are included. NARA says its site documents generally may be reproduced unless marked otherwise, while warning that some materials can contain third-party copyrighted content; this dataset uses the historical government text and paraphrased facts, and attribution is retained in the source manifest. Recheck any item-specific notice before redistributing the dataset.

## Split protocol

The document assignments were fixed first in `split_manifest.jsonl`. Question IDs, questions, conditions, source document IDs, and splits are recorded in `question_split_manifest.jsonl`. Only after these assignments were recorded were the passage cases authored. No paraphrases, alternate question wording, preference pairs, or training rows should be created from validation or test questions.

| Split | Source documents | Cases | Purpose |
| --- | --- | ---: | --- |
| train | Constitution; Articles of Confederation | 8 | Candidate source for training cases and preference pairs |
| validation | Declaration of Independence | 4 | Development and validation only |
| test | Bill of Rights | 4 | Final held-out evaluation only |

The source document is the grouping unit. Every question from a source stays in that source's assigned split; the question IDs are also unique. The current set is a tiny pilot and the source-document split is intentionally conservative. Expand by adding whole new source documents to a split before writing questions or paraphrases. Do not place the same source document in multiple splits.

## Case labels

Each row in `cases.jsonl` includes a question, fixed passages with passage IDs, passage-level labels, answerability, expected answer, and missing-information labels. The four conditions are clean, incomplete, irrelevant, and contradictory.

Contradictory cases contain one `paraphrase_of_primary_source` passage and one explicitly tagged `synthetic_counterfactual`. The counterfactual is an intentionally false perturbation used to test conflict handling; it is not a historical quotation or a claim made by the National Archives. Never re-label or cite it as a primary-source passage.

## Answer annotation

Use [`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md) to label answer claims and assess citations claim by claim. It distinguishes evidence support from citation quality, defines how to handle conflicts and missing details, and provides a JSONL annotation-record example. Keep answer annotations separate from the fixed case file.

## Training examples

The eight training cases have reviewed starter preference pairs in `preference_pairs.jsonl`. `dpo.jsonl` contains the TRL `prompt`, `chosen`, and `rejected` fields; `sft.jsonl` contains `prompt` and `completion` using the exact same prompts and chosen responses. See [`PREFERENCE_REVIEW.md`](PREFERENCE_REVIEW.md) for the manual review and the warning about current trainer defaults.

`dpo_validation.jsonl` contains one held-out preference pair for each validation case. It is used only for DPO evaluation; its chosen responses match the validation cases' expected answers. Test cases are not included in either training or validation datasets.

## Separation from the earlier prototype

The older `data/raw/prototype/source/preference_pairs.jsonl`, `data/processed/prototype/train/*.jsonl`, and `data/processed/prototype/eval/baseline.jsonl` files are synthetic pipeline prototypes. They are not part of this source-grounded split and should not be mixed with `data/processed/grounded/cases.jsonl`. Grounded SFT and DPO pairs are in this directory and are consumed by `src/train_grounded_sft.py` and `src/train_grounded_dpo.py`.
