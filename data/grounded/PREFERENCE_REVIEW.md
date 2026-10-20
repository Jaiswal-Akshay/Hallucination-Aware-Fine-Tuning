# Grounded training-pair review

## Review scope

All eight pilot preference pairs were manually reviewed against their corresponding training case in `cases.jsonl`. This author review checks evidence fidelity, citation placement, and whether chosen/rejected responses are reasonably comparable in tone and length. It is a small pilot review, not independent double annotation.

## Review notes

| Pair | Condition | Review |
| --- | --- | --- |
| grounded-001 | clean | Chosen answer matches P1. Rejected answer changes the actor and cites P1, which contradicts it. |
| grounded-002 | incomplete | Chosen answers the supported part and marks the missing count. Rejected adds a specific count absent from P1. |
| grounded-003 | irrelevant | Chosen identifies the missing date. Rejected gives a plausible date not supported by P1, which concerns congressional powers. |
| grounded-004 | contradictory | Chosen reports both sides and flags the conflict. Rejected confidently selects the explicitly labeled synthetic counterfactual P2. |
| grounded-005 | clean | Chosen matches the one-vote statement in P1. Rejected changes the number and cites P1. |
| grounded-006 | incomplete | Chosen gives the supported range and marks salary as missing. Rejected adds an unsupported salary while retaining the supported range. |
| grounded-007 | irrelevant | Chosen says the answer is absent from the supplied passage. Rejected supplies an outside identity and cites an unrelated passage. |
| grounded-008 | contradictory | Chosen reports the one-vote/two-vote conflict. Rejected confidently selects the explicitly labeled synthetic counterfactual P2. |

Chosen and rejected responses use the same prompt per pair. The rejected answers are not made longer across the board: several are shorter than their chosen counterparts, while the incomplete pairs include a similarly structured unsupported detail. Every pair still needs review for the intended audience before a substantive training run, especially because the dataset has only two cases per condition and counterfactual conflicts are synthetic.

## Files and split boundary

- `preference_pairs.jsonl`: authoring source with case IDs and rationale.
- `dpo.jsonl`: TRL DPO fields `prompt`, `chosen`, and `rejected`, derived from the authoring source.
- `sft.jsonl`: `prompt` and `completion`, using the identical prompt and chosen response for each DPO row.
- Only the eight `train` cases were used. Validation and test cases were not used to author these examples.

These files are isolated under `data/grounded/`. The existing training scripts still read the earlier synthetic prototype under `data/train/`; do not treat their current defaults as training on this grounded data. Before a grounded training run, update or add an explicit data-path option and confirm both arms load the same grounded prompts and chosen answers.
