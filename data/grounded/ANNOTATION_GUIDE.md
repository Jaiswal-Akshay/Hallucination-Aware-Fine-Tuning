# Claim and citation annotation guide

Use this guide to review model answers against the exact passages supplied with each case. Annotate what the passages establish; do not use outside knowledge to fill gaps. The goal is consistent evaluation of factual support, not judging whether a claim happens to be true in the world.

## 1. Review unit

Split an answer into its smallest independently checkable factual claims. A sentence can contain multiple claims; annotate each separately. Include factual qualifiers such as dates, quantities, actors, locations, causal relationships, and certainty. Do not score greetings, stylistic wording, or clearly marked opinions as factual claims.

For every factual claim, record:

- Exact claim text (or a concise faithful span).
- Evidence label from the four labels below.
- Citation IDs attached to that claim, if any.
- For each cited ID, whether that passage specifically supports the claim.
- A short rationale pointing to the relevant passage text or explaining the gap/conflict.

## 2. Evidence labels

Assign exactly one label to each claim, based only on the case's fixed passage set:

| Label | Apply when | Example interpretation |
| --- | --- | --- |
| **Supported** | At least one supplied passage directly entails the claim, or supports it through a clear, minimal inference. The claim does not add unsupported specifics. | Passage says Congress has the power to declare war; claim says Congress can declare war. |
| **Contradicted** | One or more supplied passages directly state an incompatible fact. Use this when the evidence explicitly conflicts with the claim, even if another supplied passage supports it. | Passage says the power belongs to Congress; claim says the President alone has that power. |
| **Unsupported** | The supplied passages neither establish nor directly contradict the claim. This includes details omitted from incomplete evidence and claims relying on outside knowledge. | Asked for a date, but no supplied passage gives the date. |
| **Too unclear to assess** | The claim or evidence is too ambiguous, incomplete, garbled, or underspecified to determine support or contradiction reliably. Do not use merely because the answer is absent; that is **Unsupported**. | “They approved it then” when the referent and time cannot be resolved from the question and passages. |

### Precedence when passages conflict

If the claim is directly supported by one supplied passage and directly contradicted by another, label it **Contradicted** and note both passage IDs. Record that the evidence set itself conflicts. Do not silently select one version or resolve the conflict using outside knowledge. If the passages are ambiguous enough that neither side can be identified reliably, use **Too unclear to assess**.

## 3. Claim-by-claim citation review

Evaluate every citation attached to every claim separately. A citation is supported only if the referenced passage backs that specific claim. A citation elsewhere in the answer does not transfer support to an uncited claim.

For each claim-citation pair, record one of:

- **Citation supports claim:** the cited passage directly supports the full claim, including its material qualifiers.
- **Citation partially supports claim:** it supports only part of a compound claim; identify the supported and unsupported spans. Split the claim into smaller claims where practical.
- **Citation contradicts claim:** the cited passage says something incompatible.
- **Citation irrelevant to claim:** it does not address the claim.
- **Citation ID invalid:** the ID is absent from that case's fixed passage list.
- **No citation:** the claim has no citation attached.

If a claim is supported by another passage but cites an irrelevant or contradictory passage, the claim's evidence label and its citation assessment can differ. Example: a claim may be **Supported** from P1 while its attached citation P2 is **Citation irrelevant to claim**. Record both facts; do not mark a bad citation as good because the answer is otherwise correct.

If a citation is attached to a sentence with multiple claims, assess which claims it actually supports. Do not assume it supports the entire sentence. A citation to a passage that supports only one clause is partial for the other clauses.

## 4. Case-condition handling

- **Clean:** expect the requested answer to be supported. Additional factual claims still need independent review.
- **Incomplete:** label supplied details that are supported normally; label omitted requested details as **Unsupported**, unless the evidence makes the issue genuinely ambiguous. An answer that clearly states the missing detail cannot be determined is not itself an unsupported factual assertion.
- **Irrelevant:** do not reward a plausible answer from memory. If no supplied passage addresses the claim, use **Unsupported** unless a passage directly contradicts it or the case is genuinely unclear.
- **Contradictory:** label claims according to the conflicting passages and apply the conflict precedence rule. The expected behavior may be to acknowledge the conflict or abstain; do not treat the synthetic counterfactual passage as historical truth. Its provenance is explicitly marked in the case.

## 5. General consistency rules

1. Use only the question and passages supplied in that case. Do not browse, use model memory, or infer beyond a clear minimal implication.
2. Compare the claim at the same level of specificity. Evidence for a broad fact does not support an added date, number, motive, or absolute qualifier.
3. A correct answer by coincidence is still **Unsupported** if the passages do not establish it.
4. A claim can receive only one evidence label. If it contains distinct propositions, split it before labeling.
5. Preserve passage IDs in rationales. Never invent or silently repair a passage or citation ID.
6. When uncertain between **Unsupported** and **Too unclear to assess**, use **Unsupported** if the claim is clear but evidence is absent; use **Too unclear to assess** only when meaning or evidence cannot be interpreted reliably.

## 6. Suggested annotation record

Keep annotations separate from the source cases so the fixed evidence and gold data remain unchanged. One JSONL row per answer is sufficient; `claims` contains one object per factual claim:

```json
{
  "case_id": "train_const_clean_01",
  "answer_id": "base_run_001",
  "annotator_id": "annotator_01",
  "claims": [
    {
      "claim_text": "Congress has the power to declare war.",
      "evidence_label": "Supported",
      "citation_ids": ["P1"],
      "citation_reviews": [
        {"passage_id": "P1", "status": "Citation supports claim"}
      ],
      "rationale": "P1 assigns the declare-war power to Congress."
    }
  ],
  "overall_notes": ""
}
```

Allowed `evidence_label` values are exactly `Supported`, `Contradicted`, `Unsupported`, and `Too unclear to assess`. Use the citation review statuses defined above. If there is no citation, set `citation_ids` to `[]` and `citation_reviews` to `[]`; the absence of a citation is still visible and should not be replaced with an inferred citation.

## 7. Adjudication

For a pilot, have a second reviewer independently annotate a subset, especially contradictory and incomplete cases. Compare disagreements by claim and passage ID. Resolve them by applying the definitions above and recording the rationale; do not change the case passages to make an answer fit. If the guide does not settle a recurring disagreement, update the guide before annotating the remaining answers, then recheck the already annotated subset.
