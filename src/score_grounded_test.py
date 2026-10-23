"""Validate manual test annotations and aggregate claim-level evaluation metrics."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "generations" / "grounded-test"
METRICS_OUT = ROOT / "outputs" / "metrics" / "grounded-test" / "metrics.json"
ANNOTATIONS = ROOT / "data" / "processed" / "grounded" / "test_annotations.jsonl"
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
    metrics["overall"] = {}
    for arm in ARMS:
        rows = [a for a in annotations if a["arm"] == arm]
        claims = [c for a in rows for c in a["claims"]]
        labels = Counter(c["evidence_label"] for c in claims)
        citation_reviews = [r for c in claims for r in c["citation_reviews"]]
        correctness = Counter(a["answer_correctness"] for a in rows)
        abstention = [a["abstention_accuracy"] for a in rows if a["abstention_accuracy"] is not None]
        covered = [a for a in rows if a["coverage"]]
        metrics["overall"][arm] = {
            "questions": len({a["case_id"] for a in rows}), "answers": len(rows),
            "factual_claims": len(claims), "claim_labels": dict(labels),
            "unsupported_claim_rate": rate(sum(labels[x] for x in ("Contradicted", "Unsupported", "Too unclear to assess")), len(claims)),
            "unsupported_only_rate": rate(labels["Unsupported"] + labels["Too unclear to assess"], len(claims)),
            "contradiction_rate": rate(labels["Contradicted"], len(claims)),
            "answers_with_any_unsupported_or_contradicted_claim": sum(any(c["evidence_label"] != "Supported" for c in a["claims"]) for a in rows),
            "correctness": dict(correctness),
            "full_correctness_rate_all_cases": rate(correctness["correct"], len(rows)),
            "full_correctness_among_covered": rate(sum(a["answer_correctness"] == "correct" for a in covered), len(covered)),
            "covered_answers": len(covered), "coverage_rate": rate(len(covered), len(rows)),
            "citation_claims": sum(bool(c["citation_reviews"]) for c in claims),
            "citation_claim_coverage": rate(sum(bool(c["citation_reviews"]) for c in claims), len(claims)),
            "citation_pairs": len(citation_reviews),
            "citation_support_rate": rate(sum(r["status"] in supported_citation_statuses for r in citation_reviews), len(citation_reviews)),
            "abstention_accuracy": rate(sum(bool(x) for x in abstention), len(abstention)),
            "abstention_correct": sum(bool(x) for x in abstention), "abstention_cases": len(abstention),
            "over_refusal_count": sum(a["over_refusal"] is True for a in rows),
            "over_refusal_cases": sum(a["over_refusal"] is not None for a in rows),
        }
    metrics["paired_comparisons"] = {
        "case_count": len(reference),
        "base_and_dpo_textually_identical": all(
            by_arm_case[("base", cid)]["generated_answer"] == by_arm_case[("dpo", cid)]["generated_answer"]
            for cid in reference
        ),
        "sft_vs_base_case_results": {
            cid: {
                "condition": by_arm_case[("base", cid)]["condition"],
                "base_correctness": next(a["answer_correctness"] for a in annotations if a["arm"] == "base" and a["case_id"] == cid),
                "sft_correctness": next(a["answer_correctness"] for a in annotations if a["arm"] == "sft" and a["case_id"] == cid),
                "base_unsupported_claim_rate": rate(sum(c["evidence_label"] != "Supported" for c in next(a for a in annotations if a["arm"] == "base" and a["case_id"] == cid)["claims"]), len(next(a for a in annotations if a["arm"] == "base" and a["case_id"] == cid)["claims"])),
                "sft_unsupported_claim_rate": rate(sum(c["evidence_label"] != "Supported" for c in next(a for a in annotations if a["arm"] == "sft" and a["case_id"] == cid)["claims"]), len(next(a for a in annotations if a["arm"] == "sft" and a["case_id"] == cid)["claims"])),
            } for cid in reference
        },
        "interpretation": "Descriptive paired outcomes only. Four questions are insufficient for inferential significance testing; claims within an answer are clustered and are not independent observations.",
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(annotations)} annotations against {len(ARMS)} arms and {len(reference)} held-out cases.")
    print(f"Wrote {METRICS_OUT}")


if __name__ == "__main__":
    main()
