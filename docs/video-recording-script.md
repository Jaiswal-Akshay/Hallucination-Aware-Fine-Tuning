# Video recording script: evidence-aware fine-tuning research prototype

**Suggested length:** 5–7 minutes  
**Audience:** classmates, project reviewers, or a research supervisor  
**Recording note:** Keep the project folder open in File Explorer and use VS Code or another editor to show the listed files. Speak the narration naturally; bracketed text is an on-screen cue, not narration.

## 1. Opening and research question — 0:00–0:40

**On screen:** Show the project title and the project folder.

**Say:**

“Hello. This project is called *Evidence-Aware Preference Optimization for Retrieval-Augmented Question Answering*. We’re studying a focused question: when an answer model receives a question and retrieved evidence, can fine-tuning encourage it to stick to what that evidence supports, identify missing information, and handle conflicts instead of guessing?

This video demonstrates a local research prototype. It simulates the evidence-answering part of a retrieval-augmented system by supplying fixed passages with each question. It is not a deployed search service, and it does not claim that fine-tuning has already solved hallucinations.”

## 2. Research setup and model — 0:40–1:20

**On screen:** Open `README.md`, then `docs/model-checkpoint.md`. Show the local model directory if useful.

**Say:**

“The controlled model for this experiment is Qwen 2.5 0.5B Instruct, loaded from a pinned Hugging Face checkpoint. The base model, SFT model, and DPO model are intended to start from the same original checkpoint, so the comparison can focus on the training method.

The laptop setup is CPU-based. That makes the experiment possible on this machine, but training is slow. We also have Llama 3 in Ollama, but it is a different model and is not part of the controlled comparison.”

## 3. How the evidence simulation works — 1:20–2:15

**On screen:** Open `data/processed/grounded/README.md`, then `source_manifest.jsonl`, `split_manifest.jsonl`, and `cases.jsonl`. Expand one case.

**Say:**

“For the source-grounded pilot, we chose a narrow topic: U.S. founding documents. Each case contains a question and a fixed set of passages, each with an ID such as P1. The model is asked to answer using only those passages and cite the passage IDs.

We include four evidence conditions: clean evidence, incomplete evidence, irrelevant evidence, and contradictory evidence. This lets us examine more than straightforward factual recall. For example, in an incomplete case, the passage may answer one part of a question but omit a requested date or number. In a contradictory case, the passages disagree. The conflicting passage marked ‘synthetic counterfactual’ is an intentional test perturbation, not a historical quotation.”

## 4. Why the split matters — 2:15–2:55

**On screen:** Show `split_manifest.jsonl` and `question_split_manifest.jsonl`; point to the split assignments.

**Say:**

“We assigned whole source documents to train, validation, or test before writing the question cases. The current pilot uses the Constitution and Articles of Confederation for training, the Declaration of Independence for validation, and the Bill of Rights for the held-out test split.

That document-level separation reduces the chance that the same source facts appear on both sides of the evaluation. The split is still small, so it is a safeguard for this pilot, not a guarantee that every possible form of leakage has been eliminated.”

## 5. How we construct and check training examples — 2:55–3:55

**On screen:** Open `data/processed/grounded/preference_pairs.jsonl`, then `data/processed/grounded/sft.jsonl` and `data/processed/grounded/dpo.jsonl`. Show one incomplete pair and one contradictory pair.

**Say:**

“For each of the eight training cases, we wrote a preferred answer and a rejected answer using the same prompt. The preferred answer states what the supplied evidence supports, cites the relevant passage IDs, and identifies information the passages do not provide. The rejected answer is written to sound plausible, but it adds an unsupported detail, conflicts with a passage, or selects one side of a conflict too confidently.

The SFT file contains each prompt with its preferred completion. The DPO file contains that same prompt, chosen answer, and rejected answer. Keeping the prompts and chosen responses aligned gives the two training approaches the same grounded target examples.

We manually reviewed all eight starter pairs for evidence fidelity, citation placement, and response comparability. This was an author review, not independent double annotation. The set is intentionally a small pipeline example and needs to grow before a strong research conclusion is possible.”

## 6. What makes the process auditable — 3:55–4:50

**On screen:** Open `data/processed/grounded/ANNOTATION_GUIDE.md`; show the four labels and citation rules. Then show `docs/training-results.md`.

**Say:**

“We make the evaluation auditable in a few ways. Each claim is judged only against the passages supplied for that case. The annotation guide distinguishes supported, contradicted, unsupported, and too unclear to assess. It also checks citations claim by claim: a citation counts only if that passage supports that specific claim.

We record source URLs and passage provenance, keep the split manifests, and preserve the fixed cases separately from answer annotations. These choices make it easier for another reviewer to trace why an answer was labeled a certain way. They do not make the model automatically trustworthy. Trust here means that the experimental process is inspectable and that uncertainty and limitations are reported honestly.”

## 7. Results and current limitations — 4:50–5:45

**On screen:** Show the results table and limitations in `docs/training-results.md`. Optionally show the grounded-data warning in README.

**Say:**

“We have already run an earlier CPU pipeline demonstration using twelve synthetic training examples and twelve synthetic evaluation examples. In that small run, SFT did not consistently prevent unsupported claims, and DPO outputs were effectively unchanged from the base model on the evaluation set. Those results do not show a hallucination reduction.

We have now trained grounded SFT and DPO adapters independently from the original Qwen checkpoint and evaluated all three arms on the same four held-out questions. Every answer was reviewed claim by claim. SFT showed fewer unsupported claims in this tiny test, while DPO outputs matched the base model. With one question per condition, this remains a pipeline pilot and does not establish that fine-tuning reduces hallucinations.”

## 8. Closing — 5:45–6:05

**On screen:** Return to project title or the four evidence conditions.

**Say:**

“The experiment now has fixed evidence, source-level splits, aligned SFT and DPO examples, independent adapters, and claim-level review of held-out answers. The next research step is to expand the source-separated dataset and test set, then repeat the paired comparison without using test results for tuning.”

## Suggested recording checklist

- Keep personal paths, tokens, and unrelated desktop notifications off screen.
- Show the `synthetic_counterfactual` label when demonstrating a contradiction case.
- Do not call the prototype a deployed RAG system or say the fine-tuned model is proven trustworthy.
- State clearly that the current test set has only four questions and results are preliminary.
- If showing model output, identify which checkpoint and dataset produced it; do not present synthetic-prototype results as grounded-corpus results.
