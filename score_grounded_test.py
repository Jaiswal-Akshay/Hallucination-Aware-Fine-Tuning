"""Validate manual test annotations and aggregate claim-level evaluation metrics."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "grounded-test-evaluation"
ANNOTATIONS = ROOT / "data" / "grounded" / "test_annotations.jsonl"
ARMS = ("base", "sft", "dpo")
LABELS = {"Supported", "Contradicted", "Unsupported", "Too unclear to assess"}


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rate(n, d):
    return None if not d else n / d


def main():
    predictions = {arm: read_jsonl(OUT / f"{arm}_predictions.jsonl") for arm in ARMS}
    annotations = read_jsonl(ANNOTATIONS)
    by_arm_case = {}
    reference = None
    for arm, rows in predictions.items():
        ids = [r["case_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate prediction case IDs for {arm}")
        if reference is None:
            reference = {r["case_id"]: r["prompt"] for r in rows}
        elif {r["case_id"]: r["prompt"] for r in rows} != reference:
            raise ValueError(f"Test cases/prompts differ for arm {arm}")
        for row in rows:
            by_arm_case[(arm, row["case_id"])] = row
    if len(annotations) != len(ARMS) * len(reference):
        raise ValueError(f"Expected {len(ARMS) * len(reference)} annotations, got {len(annotations)}")
    seen = set()
    supported_citation_statuses = {"Citation supports claim"}
    metrics = {"definitions": {
        "unsupported_claim_rate": "(Contradicted + Unsupported + Too unclear to assess claims) / all factual claims; contradiction is also reported separately and overlaps this numerator.",
        "contradiction_rate": "Contradicted claims / all factual claims.",
        "citation_support_rate": "Citation-claim pairs marked Citation supports claim / all reviewed citation-claim pairs; null when no citations are present.",
        "citation_claim_coverage": "Factual claims with at least one reviewed citation / all factual claims.",
        "full_correctness_rate": "Answers marked fully correct / answers in condition; partial is reported separately.",
        "coverage": "Answers that provide a substantive answer or evidence-grounded qualification / all answers.",
        "abstention_accuracy": "Correct abstention/qualification decisions / applicable cases. Clean cases are not applicable.",
    }, "by_arm_condition": {}}
    for a in annotations:
        key = (a["arm"], a["case_id"])
        if key in seen:
            raise ValueError(f"Duplicate annotation: {key}")
        seen.add(key)
        if key not in by_arm_case:
            raise ValueError(f"Annotation has no matching prediction: {key}")
        pred = by_arm_case[key]
        if a["answer_id"] != pred["answer_id"] or a["condition"] != pred["condition"]:
            raise ValueError(f"Answer ID/condition mismatch: {key}")
        for claim in a["claims"]:
            if claim["evidence_label"] not in LABELS:
                raise ValueError(f"Invalid evidence label in {key}: {claim['evidence_label']}")
            if len(claim["citation_ids"]) != len(claim["citation_reviews"]):
                raise ValueError(f"Citation review count mismatch in {key}")
            for cid, review in zip(claim["citation_ids"], claim["citation_reviews"]):
                if cid != review["passage_id"]:
                    raise ValueError(f"Citation ID mismatch in {key}")
                if review["status"] not in {"Citation supports claim", "Citation partially supports claim", "Citation contradicts claim", "Citation irrelevant to claim", "Citation ID invalid"}:
                    raise ValueError(f"Invalid citation status in {key}: {review['status']}")
    expected = {(arm, case_id) for arm in ARMS for case_id in reference}
    if seen != expected:
        raise ValueError(f"Annotation coverage mismatch: missing={expected-seen}, extra={seen-expected}")

    conditions = ("clean", "incomplete", "irrelevant", "contradictory")
    for arm in ARMS:
        metrics["by_arm_condition"][arm] = {}
        for condition in conditions:
            rows = [a for a in annotations if a["arm"] == arm and a["condition"] == condition]
            claims = [c for a in rows for c in a["claims"]]
            labels = Counter(c["evidence_label"] for c in claims)
            cite_reviews = [r for c in claims for r in c["citation_reviews"]]
            cited_claims = sum(bool(c["citation_reviews"]) for c in claims)
            applicable_abstention = [a["abstention_accuracy"] for a in rows if a["abstention_accuracy"] is not None]
            correctness = Counter(a["answer_correctness"] for a in rows)
            factual = len(claims)
            metrics["by_arm_condition"][arm][condition] = {
                "answers": len(rows), "factual_claims": factual,
                "claim_labels": dict(labels),
                "unsupported_claim_rate": rate(sum(labels[x] for x in ("Contradicted", "Unsupported", "Too unclear to assess")), factual),
                "unsupported_only_rate": rate(labels["Unsupported"] + labels["Too unclear to assess"], factual),
                "contradiction_rate": rate(labels["Contradicted"], factual),
                "correctness": dict(correctness),
                "full_correctness_rate": rate(correctness["correct"], len(rows)),
                "covered_answers": sum(bool(a["coverage"]) for a in rows),
                "full_correctness_among_covered": rate(
                    sum(a["answer_correctness"] == "correct" and a["coverage"] for a in rows),
                    sum(bool(a["coverage"]) for a in rows),
                ),
                "citation_claims": cited_claims, "citation_claim_coverage": rate(cited_claims, factual),
                "citation_pairs": len(cite_reviews),
                "citation_support_rate": rate(sum(r["status"] in supported_citation_statuses for r in cite_reviews), len(cite_reviews)),
                "citation_statuses": dict(Counter(r["status"] for r in cite_reviews)),
                "abstention_accuracy": rate(sum(bool(v) for v in applicable_abstention), len(applicable_abstention)),
                "abstention_cases": len(applicable_abstention),
                "over_refusal_count": sum(bool(a["over_refusal"]) for a in rows if a["over_refusal"] is not None),
                "over_refusal_cases": sum(a["over_refusal"] is not None for a in rows),
                "coverage": rate(sum(bool(a["coverage"]) for a in rows), len(rows)),
            }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(annotations)} annotations against {len(ARMS)} arms and {len(reference)} held-out cases.")
    print(f"Wrote {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
