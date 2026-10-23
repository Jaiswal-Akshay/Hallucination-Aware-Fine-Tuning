# CPU adapter training results

## Run configuration

Both adapters used `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`, LoRA rank 4, sequence length 256, batch size 1, and one epoch over the same 12 training prompts. Each trainer loaded the original local base checkpoint. DPO did not load the SFT adapter. Both use CPU float32.

| Arm | Training loss | Training time | Output |
| --- | ---: | ---: | --- |
| Base | — | — | `outputs/generations/baseline/` |
| SFT | 1.23 | 1,163 sec (19m 23s) | `outputs/adapters/sft-prototype/` |
| DPO | 0.6984 | 2,802 sec (46m 42s) | `outputs/adapters/dpo-prototype/` |

The DPO run spent about 77 seconds precomputing reference log probabilities; its 12 optimizer steps averaged about 234 seconds each.

## Held-out evaluation observations

All three arms were evaluated on the same 12 examples in `data/processed/prototype/eval/baseline.jsonl`.

- **Base:** Correct answer-key terms appeared for clean examples, but citations were omitted; the model invented a launch year, added an unsupported cost claim, and selected one side in each contradictory case.
- **SFT:** Generated a citation for two of three clean examples, but retained the invented launch year, unsupported cost claim, and conflict-selection failures.
- **DPO:** Predictions were effectively unchanged from base on this set; citations were still omitted and the same unsupported and contradictory-evidence failures remained.

These results do **not** show that DPO or SFT reduced hallucinations. The training set contains only 12 synthetic examples and the evaluation set only 12 examples. The losses are training diagnostics, not outcome measures. Use manual claim-level review; keyword overlap and citation-ID coverage are simple proxies only.

## Output files

- Base predictions: `outputs/generations/baseline/`; proxy metrics: `outputs/metrics/baseline/`
- SFT adapter and training summary: `outputs/adapters/sft-prototype/`
- SFT predictions: `outputs/generations/sft-prototype/`; proxy metrics: `outputs/metrics/sft-prototype/`
- DPO adapter and training summary: `outputs/adapters/dpo-prototype/`
- DPO predictions: `outputs/generations/dpo-prototype/`; proxy metrics: `outputs/metrics/dpo-prototype/`

## Next work

Review and substantially expand the training pairs and held-out evaluation set, keeping them disjoint. Then repeat training and the same evaluation protocol. For a useful research comparison, move DPO training to a supported GPU environment; this laptop took nearly 47 minutes for only 12 pairs and one epoch.
