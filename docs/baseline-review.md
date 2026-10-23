# Base-model baseline review

Run: `python evaluate_base.py`  
Checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`  
Predictions: `outputs/generations/baseline/predictions.jsonl`  
Proxy metrics: `outputs/metrics/baseline/metrics.json`

## Observations from the 12-example starter set

- **Clean evidence:** The model included the answer-key facts in all three answers, but omitted every requested passage citation.
- **Incomplete evidence:** It handled missing details in two answers, but one answer changed “no cost is listed” into “there was no shipping cost,” which is an unsupported claim. It omitted requested citations in all three answers.
- **Irrelevant evidence:** It invented a 2018 launch year for the Helio sensor in one case. It abstained on the other two questions.
- **Contradictory evidence:** It selected an answer rather than preserving the conflict in all three cases. In one answer it mentioned both sources and then chose one; in another, it incorrectly said both sources supported 120.

## Interpretation

This is a small smoke-test set, not a publishable benchmark. The outputs show useful failure cases for the project: unsupported details under incomplete or irrelevant evidence, mishandled conflicts, and missing citations. The generated `metrics.json` reports simple answer-key overlap and required-citation coverage only. These proxies do not score semantic correctness, claim-level support, or abstention quality. Manually review predictions before drawing conclusions and keep this evaluation file out of all training data.
